import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.model_v3 import Brittain3, Brittain3Config


def tiny_config(**changes):
    values = dict(
        vocab_size=128, max_seq_len=128, n_layer=2, n_head=4,
        n_kv_head=2, n_embd=32, intermediate_size=64,
        activation_checkpointing=False, logit_chunk_size=8,
    )
    values.update(changes)
    return Brittain3Config(**values)


def test_production_parameter_count_and_shape():
    model = Brittain3(Brittain3Config())
    assert model.num_params() == 181_529_216
    assert model.cfg.head_dim == 64
    assert model.cfg.n_head // model.cfg.n_kv_head == 2


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"n_embd": 30}, "divisible"),
        ({"n_kv_head": 3}, "divisible"),
        ({"n_embd": 28}, "even"),
        ({"max_seq_len": 0}, "positive"),
        ({"bias": True}, "does not support"),
    ],
)
def test_invalid_configurations_fail_clearly(changes, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**changes)


def test_forward_backward_and_chunked_loss_match():
    torch.manual_seed(1)
    model = Brittain3(tiny_config(activation_checkpointing=True))
    model.train()
    inputs = torch.randint(0, model.cfg.vocab_size, (2, 24))
    targets = torch.randint(0, model.cfg.vocab_size, (2, 24))
    targets[:, :3] = -100
    logits, full_loss = model(inputs, targets)
    _, chunked_loss = model(inputs, targets, return_logits=False)
    assert logits.shape == (2, 24, model.cfg.vocab_size)
    torch.testing.assert_close(full_loss, chunked_loss, rtol=1e-6, atol=1e-6)
    chunked_loss.backward()
    assert model.tok_emb.weight.grad is not None


def test_cached_and_uncached_logits_match_and_cache_stays_grouped():
    torch.manual_seed(2)
    model = Brittain3(tiny_config()).eval()
    inputs = torch.randint(0, model.cfg.vocab_size, (1, 32))
    full, _ = model(inputs)
    caches = [{} for _ in model.blocks]
    model(inputs[:, :-1], caches=caches)
    cached, _ = model(inputs[:, -1:], caches=caches, offset=31)
    torch.testing.assert_close(full, cached, rtol=1e-5, atol=1e-6)
    assert caches[0]["k"].shape == (1, model.cfg.n_kv_head, 32, model.cfg.head_dim)
    assert caches[0]["v"].shape == caches[0]["k"].shape


def test_stream_cached_and_uncached_tokens_match_with_greedy_sampling():
    torch.manual_seed(3)
    model = Brittain3(tiny_config()).eval()
    inputs = torch.randint(0, model.cfg.vocab_size, (1, 12))
    cached = list(model.stream(inputs, 5, top_k=1, use_cache=True))
    uncached = list(model.stream(inputs, 5, top_k=1, use_cache=False))
    assert [item.item() for item in cached] == [item.item() for item in uncached]


@pytest.mark.parametrize("context", [1024, 2048, 4096, 8192, 16384])
def test_all_supported_contexts_complete_forward_backward_without_materialized_attention(
    context, monkeypatch
):
    calls = []

    def linear_attention(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, kwargs))
        # Preserve shape and a gradient path without allocating T by T scores.
        return q + v.mean(dim=2, keepdim=True)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", linear_attention)
    cfg = Brittain3Config(
        vocab_size=32, max_seq_len=16384, n_layer=1, n_head=1,
        n_kv_head=1, n_embd=8, intermediate_size=16,
        activation_checkpointing=True, logit_chunk_size=256,
    )
    model = Brittain3(cfg).train()
    inputs = torch.zeros((1, context), dtype=torch.long)
    targets = torch.ones((1, context), dtype=torch.long)
    _, loss = model(inputs, targets, return_logits=False)
    loss.backward()
    assert calls and calls[0][0][-2] == context


@pytest.mark.skipif(
    os.environ.get("BRITTAIN_RUN_REAL_16K") != "1",
    reason="set BRITTAIN_RUN_REAL_16K=1 for the fused 16K hardware test",
)
def test_real_fused_16k_forward_backward():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    cfg = Brittain3Config(
        vocab_size=32, max_seq_len=16384, n_layer=1, n_head=1,
        n_kv_head=1, n_embd=8, intermediate_size=16,
        activation_checkpointing=True, logit_chunk_size=256,
    )
    model = Brittain3(cfg).to(device).train()
    inputs = torch.zeros((1, 16384), dtype=torch.long, device=device)
    targets = torch.ones_like(inputs)
    _, loss = model(inputs, targets, return_logits=False)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.skipif(
    os.environ.get("BRITTAIN_RUN_REAL_16K") != "1",
    reason="set BRITTAIN_RUN_REAL_16K=1 for the fused 16K hardware test",
)
def test_real_fused_16k_cache_equivalence():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    cfg = Brittain3Config(
        vocab_size=32, max_seq_len=16384, n_layer=1, n_head=1,
        n_kv_head=1, n_embd=8, intermediate_size=16,
        activation_checkpointing=False,
    )
    model = Brittain3(cfg).to(device).eval()
    inputs = torch.arange(16384, device=device).remainder(cfg.vocab_size).view(1, -1)
    full, _ = model(inputs)
    caches = [{} for _ in model.blocks]
    model(inputs[:, :-1], caches=caches)
    cached, _ = model(inputs[:, -1:], caches=caches, offset=16383)
    torch.testing.assert_close(full, cached, rtol=1e-4, atol=1e-5)
    assert caches[0]["k"].shape == (1, 1, 16384, cfg.head_dim)
