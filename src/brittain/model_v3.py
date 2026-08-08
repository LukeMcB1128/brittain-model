"""Brittain3 decoder-only Transformer.

This module is separate from ``model.py`` so Brittain1 and Brittain2 checkpoint
loading stays unchanged. Brittain3 uses grouped-query attention, RMSNorm, QK
normalization, and a fixed 16K RoPE domain.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


BRITTAIN3_ARCHITECTURE = "brittain3"
BRITTAIN3_ARCHITECTURE_VERSION = 1


@dataclass
class Brittain3Config:
    """Architecture-only configuration for a Brittain3 checkpoint."""

    vocab_size: int = 24_576
    max_seq_len: int = 16_384
    n_layer: int = 18
    n_head: int = 14
    n_kv_head: int = 7
    n_embd: int = 896
    intermediate_size: int = 2_400
    rope_theta: float = 100_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    bias: bool = False
    qk_norm: bool = True
    activation_checkpointing: bool = True
    logit_chunk_size: int = 256
    architecture: str = BRITTAIN3_ARCHITECTURE
    architecture_version: int = BRITTAIN3_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        if self.architecture != BRITTAIN3_ARCHITECTURE:
            raise ValueError(f"architecture must be {BRITTAIN3_ARCHITECTURE!r}")
        if self.architecture_version != BRITTAIN3_ARCHITECTURE_VERSION:
            raise ValueError(
                f"unsupported Brittain3 architecture version {self.architecture_version}"
            )
        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "n_embd": self.n_embd,
            "intermediate_size": self.intermediate_size,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("the attention head dimension must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 1.0:
            raise ValueError("rope_theta must be greater than 1")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.logit_chunk_size < 0:
            raise ValueError("logit_chunk_size cannot be negative")
        if self.bias:
            raise ValueError("Brittain3 does not support linear biases")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (value.to(source_dtype) * self.weight.to(source_dtype))


def build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build float32 split-half RoPE tables."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Apply split-half RoPE to ``(B, H, T, D)`` input."""
    length = x.size(-2)
    local_cos = cos[offset:offset + length].to(dtype=x.dtype).view(1, 1, length, -1)
    local_sin = sin[offset:offset + length].to(dtype=x.dtype).view(1, 1, length, -1)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * local_cos + rotated * local_sin


def repeat_kv(value: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand compact KV heads for PyTorch versions without SDPA GQA support."""
    if groups == 1:
        return value
    return value.repeat_interleave(groups, dim=1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: Brittain3Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.groups = cfg.n_head // cfg.n_kv_head
        self.dropout = cfg.dropout
        kv_width = cfg.n_kv_head * self.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin, offset)
        k = apply_rope(self.k_norm(k), cos, sin, offset)

        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v

        full_k = repeat_kv(k, self.groups)
        full_v = repeat_kv(v, self.groups)
        if q.size(2) == full_k.size(2):
            result = F.scaled_dot_product_attention(
                q,
                full_k,
                full_v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif q.size(2) == 1:
            result = F.scaled_dot_product_attention(q, full_k, full_v, dropout_p=0.0)
        else:
            raise ValueError(
                "Brittain3 cache updates must contain one token; partial cached "
                "prefill is disabled so no explicit attention matrix is allocated"
            )
        result = result.transpose(1, 2).contiguous().view(batch, length, width)
        return self.o_proj(result)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Brittain3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Brittain3Block(nn.Module):
    def __init__(self, cfg: Brittain3Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, offset)
        return x + self.mlp(self.ffn_norm(x))


class Brittain3(nn.Module):
    """Brittain3 causal language model with compact KV-cache storage."""

    def __init__(self, cfg: Brittain3Config):
        super().__init__()
        cfg.__post_init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Brittain3Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._rope_cache: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(
                    parameter, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer)
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "architecture": BRITTAIN3_ARCHITECTURE,
            "architecture_version": BRITTAIN3_ARCHITECTURE_VERSION,
            "parameter_count": self.num_params(),
            "max_context": self.cfg.max_seq_len,
        }

    def _rope(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if length > self.cfg.max_seq_len:
            raise ValueError(
                f"requested position {length} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        key = (device.type, device.index)
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, device, self.cfg.rope_theta
            )
        return self._rope_cache[key]

    def _run_blocks(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        caches: list[dict[str, torch.Tensor]] | None,
        offset: int,
    ) -> torch.Tensor:
        use_checkpoint = (
            self.training
            and self.cfg.activation_checkpointing
            and caches is None
            and torch.is_grad_enabled()
        )
        for index, block in enumerate(self.blocks):
            cache = None if caches is None else caches[index]
            if use_checkpoint:
                x = checkpoint(
                    lambda value, c=cos, s=sin, layer=block, off=offset: layer(
                        value, c, s, None, off
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, cos, sin, cache, offset)
        return self.final_norm(x)

    def _chunked_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        chunk = self.cfg.logit_chunk_size or hidden.size(1)
        total = hidden.new_zeros((), dtype=torch.float32)
        valid = 0

        def chunk_loss(local_hidden: torch.Tensor, local_targets: torch.Tensor) -> torch.Tensor:
            local_logits = self.lm_head(local_hidden)
            return F.cross_entropy(
                local_logits.reshape(-1, local_logits.size(-1)),
                local_targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).float()

        for start in range(0, hidden.size(1), chunk):
            stop = min(start + chunk, hidden.size(1))
            local_targets = targets[:, start:stop]
            local_valid = int((local_targets != -100).sum().item())
            if not local_valid:
                continue
            local_hidden = hidden[:, start:stop]
            if self.training and torch.is_grad_enabled():
                # Recompute the vocabulary projection during backward. This keeps
                # every chunk's large logits out of the retained forward graph.
                local_loss = checkpoint(
                    chunk_loss, local_hidden, local_targets, use_reentrant=False
                )
            else:
                local_loss = chunk_loss(local_hidden, local_targets)
            total = total + local_loss
            valid += local_valid
        if not valid:
            raise ValueError("targets contain no graded tokens")
        return total / valid

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[dict[str, torch.Tensor]] | None = None,
        offset: int = 0,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        if idx.size(1) < 1:
            raise ValueError("idx must contain at least one token")
        if offset < 0 or offset + idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("the requested token positions are outside max_seq_len")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("caches must contain one dictionary per layer")
        cos, sin = self._rope(max(offset + idx.size(1), 1), idx.device)
        hidden = self.dropout(self.tok_emb(idx))
        hidden = self._run_blocks(hidden, cos, sin, caches, offset)
        if targets is None:
            return self.lm_head(hidden[:, [-1]]), None
        if targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        if return_logits:
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return logits, loss
        return None, self._chunked_loss(hidden, targets)

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        history: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
    ) -> torch.Tensor:
        scores = logits[:, -1].float().clone()
        if repetition_penalty != 1.0:
            for batch in range(history.size(0)):
                prior = torch.unique(history[batch])
                selected = scores[batch, prior]
                scores[batch, prior] = torch.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )
        scores /= max(temperature, 1e-5)
        if top_k is not None:
            threshold = torch.topk(scores, min(top_k, scores.size(-1))).values[:, [-1]]
            scores = scores.masked_fill(scores < threshold, -float("inf"))
        if top_p is not None:
            sorted_scores, sorted_ids = torch.sort(scores, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            for batch in range(scores.size(0)):
                scores[batch, sorted_ids[batch, remove[batch]]] = -float("inf")
        return torch.multinomial(F.softmax(scores, dim=-1), 1)

    @torch.no_grad()
    def stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.4,
        top_k: int | None = None,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.12,
        use_cache: bool = True,
    ):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if not use_cache:
            history = idx
            for _ in range(max_new_tokens):
                logits, _ = self(history[:, -self.cfg.max_seq_len:])
                token = self._sample(
                    logits, history, temperature, top_k, top_p, repetition_penalty
                )
                history = torch.cat((history, token), dim=1)
                yield token
            return

        history = idx
        caches: list[dict[str, torch.Tensor]] = [{} for _ in self.blocks]
        context = history[:, -self.cfg.max_seq_len:]
        logits, _ = self(context, caches=caches)
        cached = context.size(1)
        for _ in range(max_new_tokens):
            token = self._sample(
                logits, history, temperature, top_k, top_p, repetition_penalty
            )
            history = torch.cat((history, token), dim=1)
            yield token
            if cached >= self.cfg.max_seq_len:
                caches = [{} for _ in self.blocks]
                context = history[:, -(self.cfg.max_seq_len - 1):]
                logits, _ = self(context, caches=caches)
                cached = context.size(1)
            else:
                logits, _ = self(token, caches=caches, offset=cached)
                cached += 1

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        result = idx
        for token in self.stream(idx, max_new_tokens, **kwargs):
            result = torch.cat((result, token), dim=1)
        return result
