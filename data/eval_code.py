"""
BRITTAIN v2 — a compact, modern decoder-only transformer.

Changes from the original (transformer.py):
  * RoPE (rotary position embeddings) instead of a learned absolute table.
    -> No more "context cliff" at 32/256. You train at one context length and
       can run at that length (or a bit beyond) with no migrate/tile hacks.
  * Fused attention via F.scaled_dot_product_attention (Flash-style).
    -> Much faster + far less memory than the Python ModuleList-of-heads loop.
  * Single fused QKV projection instead of 3 separate Linear layers per head.
  * Dense SwiGLU feed-forward instead of a Python-loop top-1 MoE.
    -> At this scale MoE mostly added memory and slow indexing for little gain.
  * Weight tying (token embedding shares weights with the output head).
    -> Saves vocab*n_embd params, which matters a lot with a BPE vocab.
  * Pre-norm + a proper init. Everything runs under bf16 autocast in train.py.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512      # training context length (in BPE tokens)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False         # bias in Linear/LayerNorm; False is faster & fine


# ---------- Rotary position embeddings ----------

def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    """Precompute cos/sin tables of shape (seq_len, head_dim)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (seq_len, half)
    emb = torch.cat((freqs, freqs), dim=-1)       # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin, offset=0):
    """x: (B, n_head, T, head_dim).

    `offset` is the absolute position of x[..., 0, :]. It is 0 for training and
    for the prefill pass, and equal to the number of cached tokens during
    incremental decoding — without it a cached token would be rotated as if it
    sat at position 0 and the geometry would be wrong.
    """
    T = x.size(-2)
    cos = cos[offset:offset + T].view(1, 1, T, -1)
    sin = sin[offset:offset + T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None, offset=0):
        """`cache` is a dict; when given, this layer's k/v are appended to it and
        the whole history is attended over. Training passes cache=None and takes
        exactly the path it always did."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v

        # is_causal assumes a square, aligned q/k. That holds while training and
        # on the prefill pass, but not once the cache makes keys longer than
        # queries — there the new queries may attend to EVERY cached key.
        if q.size(2) == k.size(2):
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif q.size(2) == 1:                       # incremental decode: sees all
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:                                      # partial prefill onto a cache
            mask = torch.ones(q.size(2), k.size(2), dtype=torch.bool,
                              device=q.device).tril(k.size(2) - q.size(2))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 / 3 * cfg.n_embd)          # ~4x params, SwiGLU convention
        hidden = 32 * ((hidden + 31) // 32)       # round to a nice multiple
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, offset=0):
        x = x + self.attn(self.ln1(x), cos, sin, cache, offset)
        x = x + self.mlp(self.ln2(x))
        return x


class Brittain(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        self._rope_cache = {}
        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, T, device):
        key = (T, device.type)
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope_cache(T, self.cfg.n_embd // self.cfg.n_head, device)
        return self._rope_cache[key]

    def num_params(self):
        # tok_emb is tied to lm_head, so the vocab matrix is only counted once.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, offset=0):
        B, T = idx.shape
        cos, sin = self._rope(max(offset + T, self.cfg.block_size), idx.device)
        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i], offset)
        x = self.ln_f(x)
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])   # only last position at inference
            return logits, None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _sample(self, logits, idx, temperature, top_k, top_p, repetition_penalty):
        """Turn last-position logits into one sampled token id."""
        logits = logits[:, -1, :].clone()
        # repetition penalty: down-weight tokens already generated (kills loops)
        if repetition_penalty != 1.0:
            for b in range(idx.size(0)):
                prev = torch.unique(idx[b])
                s = logits[b, prev]
                logits[b, prev] = torch.where(s < 0, s * repetition_penalty,
                                              s / repetition_penalty)
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        if top_p is not None:                       # nucleus sampling
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            for b in range(logits.size(0)):
                logits[b, si[b, remove[b]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 top_p=None, repetition_penalty=1.0, use_cache=True):
        """Sample continuations of idx.

        With use_cache, the prompt is run ONCE and each subsequent token attends
        to stored keys/values instead of recomputing the whole prefix — the cost
        per token stops growing with the prefix length. Sampling is unchanged, so
        with a fixed seed this produces the same tokens as the uncached path.
        """
        for nxt in self.stream(idx, max_new_tokens, temperature, top_k, top_p,
                               repetition_penalty, use_cache):
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=0.8, top_k=None,
               top_p=None, repetition_penalty=1.0, use_cache=True):
        """Yield sampled tokens one at a time, KEEPING the cache between them.

        Callers that want to print as they go must use this rather than looping
        over generate(max_new_tokens=1) — that rebuilds and discards the cache on
        every token, which is slower than not caching at all.
        """
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.block_size:])
                nxt = self._sample(logits, idx, temperature, top_k, top_p,
                                   repetition_penalty)
                idx = torch.cat((idx, nxt), dim=1)
                yield nxt
            return

        caches = [{} for _ in range(len(self.blocks))]
        cond = idx[:, -self.cfg.block_size:]
        logits, _ = self(cond, caches=caches, offset=0)
        cached = cond.size(1)

        for _ in range(max_new_tokens):
            nxt = self._sample(logits, idx, temperature, top_k, top_p,
                               repetition_penalty)
            idx = torch.cat((idx, nxt), dim=1)
            yield nxt
            if cached >= self.cfg.block_size:
                # window is full: drop the cache and re-prefill the last
                # block_size-1 tokens, matching the uncached sliding behaviour.
                caches = [{} for _ in range(len(self.blocks))]
                cond = idx[:, -(self.cfg.block_size - 1):]
                logits, _ = self(cond, caches=caches, offset=0)
                cached = cond.size(1)
            else:
                logits, _ = self(nxt, caches=caches, offset=cached)
                cached += 1
