"""
Loader for the model trained by train_50m.bs (brittain2-xs-coder:50m).

That checkpoint is a bare `nn.ModuleList.state_dict()` and its architecture is
NOT the one in model.py — BrittainScript couldn't express RoPE, SwiGLU, or weight
tying, so the .bs model uses learned position embeddings, a GELU MLP, and an
untied head. This module rebuilds that exact layout so the checkpoint loads.

Module layout written by train_50m.bs:
    0                 token embedding
    1                 position embedding
    2 + b*6 + 0..5    block b: ln1, qkv, proj, ln2, fc, fcproj
    2 + n_layer*6     final layernorm
    3 + n_layer*6     lm head

    PYTHONPATH=src python3 -m brittain.model_bs checkpoints/brittain_50m_bs.pt
"""
import sys
import codecs

import torch
import torch.nn as nn
from torch.nn import functional as F

from .paths import CHECKPOINT_DIR
from .tokenizer import CodeTok


class BrittainBS(nn.Module):
    def __init__(self, vocab=32000, n_layer=6, n_head=8, n_embd=512, block=512):
        super().__init__()
        self.vocab, self.n_layer = vocab, n_layer
        self.n_head, self.n_embd, self.block = n_head, n_embd, block
        self.head_dim = n_embd // n_head
        m = [nn.Embedding(vocab, n_embd), nn.Embedding(block, n_embd)]
        for _ in range(n_layer):
            m += [nn.LayerNorm(n_embd),
                  nn.Linear(n_embd, 3 * n_embd, bias=False),
                  nn.Linear(n_embd, n_embd, bias=False),
                  nn.LayerNorm(n_embd),
                  nn.Linear(n_embd, 4 * n_embd, bias=False),
                  nn.Linear(4 * n_embd, n_embd, bias=False)]
        m += [nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab, bias=False)]
        self.mods = nn.ModuleList(m)

    def forward(self, idx, caches=None, offset=0):
        """`caches` is a list of per-layer dicts; `offset` is the absolute
        position of idx[:, 0].

        The offset matters more here than it would with RoPE. This model uses
        LEARNED position embeddings, so position is looked up by index — a cached
        token fed at offset 0 would be told it is at the start of the sequence.
        """
        B, T = idx.shape
        dev = idx.device
        x = self.mods[0](idx) + self.mods[1](torch.arange(offset, offset + T,
                                                          device=dev))
        for b in range(self.n_layer):
            base = 2 + b * 6
            qkv = self.mods[base + 1](self.mods[base](x))
            q, k, v = qkv.chunk(3, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            if caches is not None:
                c = caches[b]
                if "k" in c:
                    k = torch.cat((c["k"], k), dim=2)
                    v = torch.cat((c["v"], v), dim=2)
                c["k"], c["v"] = k, v

            # The plain tril mask assumes a square, aligned q/k. That stops being
            # true once the cache makes keys outnumber queries: one new query may
            # attend to EVERY cached key. tril(k_len - q_len) is the general form
            # and collapses to the original mask when the two are equal.
            kt = k.size(2)
            mask = torch.ones(T, kt, device=dev, dtype=torch.bool).tril(kt - T)
            att = F.scaled_dot_product_attention(q, k, v, mask)
            att = att.transpose(1, 2).contiguous().view(B, T, self.n_embd)
            x = x + self.mods[base + 2](att)
            h = F.gelu(self.mods[base + 4](self.mods[base + 3](x)))
            x = x + self.mods[base + 5](h)
        x = self.mods[2 + self.n_layer * 6](x)
        return self.mods[3 + self.n_layer * 6](x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _sample(self, logits, idx, temperature, top_p, repetition_penalty):
        """Last-position logits -> one token id. Unchanged sampling, so a cached
        run reproduces an uncached one exactly."""
        logits = logits[:, -1, :].clone()
        # Code legitimately repeats (identifiers, indentation), so the penalty is
        # mild — but 1.0 plus a low temperature makes small models loop forever.
        if repetition_penalty != 1.0:
            for b in range(idx.size(0)):
                prev = torch.unique(idx[b])
                sc = logits[b, prev]
                logits[b, prev] = torch.where(sc < 0, sc * repetition_penalty,
                                              sc / repetition_penalty)
        logits = logits / max(temperature, 1e-5)
        if top_p is not None:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            drop = cum > top_p
            drop[..., 1:] = drop[..., :-1].clone()
            drop[..., 0] = False
            for b in range(logits.size(0)):
                logits[b, si[b, drop[b]]] = -float("inf")
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.4, top_p=0.95,
                 repetition_penalty=1.12, use_cache=True):
        for nxt in self.stream(idx, max_new_tokens, temperature, top_p,
                               repetition_penalty, use_cache):
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=0.4, top_p=0.95,
               repetition_penalty=1.12, use_cache=True):
        """Yield tokens one at a time, KEEPING the cache between them.

        Without this every token ran a full forward over the whole prefix, which
        made this 52M model slower than the 235M — that one has a cache and this
        one did not. Callers that print as they go must use stream() rather than
        looping over generate(1), which rebuilds the cache per token and is worse
        than no cache at all.

        The window slides at block_size because the POSITION TABLE only has
        block_size rows; unlike RoPE there is no extrapolating past it. At that
        point the cache is dropped and the last block-1 tokens are re-prefilled
        from position 0, which is what the old idx[:, -block:] slice did.

        EXACT within the window, DIVERGENT past it — and unavoidably so. Up to
        block_size the cached and uncached paths agree token for token. After the
        slide they do not, because these are LEARNED ABSOLUTE positions: dropping
        the oldest token shifts every remaining token's position index by one, so
        the two paths are asking the model slightly different questions. model.py
        does not have this problem, since RoPE encodes relative distance and a
        re-prefill preserves it. Neither path is more correct here; both are
        sliding-window heuristics past a context the model cannot represent.
        Generations up to block_size are bit-identical, which covers most use.
        """
        if not use_cache:
            for _ in range(max_new_tokens):
                logits = self(idx[:, -self.block:])
                nxt = self._sample(logits, idx, temperature, top_p,
                                   repetition_penalty)
                idx = torch.cat((idx, nxt), dim=1)
                yield nxt
            return

        caches = [{} for _ in range(self.n_layer)]
        cond = idx[:, -self.block:]
        logits = self(cond, caches=caches, offset=0)
        cached = cond.size(1)

        for _ in range(max_new_tokens):
            nxt = self._sample(logits, idx, temperature, top_p, repetition_penalty)
            idx = torch.cat((idx, nxt), dim=1)
            yield nxt
            if cached >= self.block:
                caches = [{} for _ in range(self.n_layer)]
                cond = idx[:, -(self.block - 1):]
                logits = self(cond, caches=caches, offset=0)
                cached = cond.size(1)
            else:
                logits = self(nxt, caches=caches, offset=cached)
                cached += 1


def load(path, device):
    model = BrittainBS().to(device)
    sd = torch.load(path, map_location=device)
    # train_50m.bs saves the ModuleList's own state_dict, so keys are "0.weight",
    # not "mods.0.weight" — load into the ModuleList directly.
    if not any(k.startswith("mods.") for k in sd):
        model.mods.load_state_dict(sd)
    else:
        model.load_state_dict(sd)
    model.eval()
    return model, CodeTok()


if __name__ == "__main__":
    ckpt = (sys.argv[1] if len(sys.argv) > 1
            else str(CHECKPOINT_DIR / "brittain_50m_bs.pt"))
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model, enc = load(ckpt, device)
    print(f"Loaded {ckpt} ({model.num_params():,} params) — trained by BrittainScript")
    print("-" * 60)
    while True:
        try:
            prompt = input("\nPrompt: ")
            if not prompt.strip():
                continue
            ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
            utf8 = codecs.getincrementaldecoder("utf-8")("replace")
            print(prompt, end="", flush=True)
            # stream(), not generate(1) in a loop — the latter rebuilds the cache
            # for every token, which is slower than not caching at all.
            for tok in model.stream(ids, 200):
                nxt = tok[0, -1].item()
                if nxt == enc.eot:
                    break
                print(utf8.decode(enc.token_bytes(nxt)), end="", flush=True)
            print("\n" + "-" * 40)
        except KeyboardInterrupt:
            print("\nbye")
            break
