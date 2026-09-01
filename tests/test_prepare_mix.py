import importlib.util
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "prepare_shakespeare", PROJECT_ROOT / "scripts/prepare/prepare_shakespeare.py"
)
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)

from brittain.data_story import EncodedStory


def segment(length, source):
    return EncodedStory(
        ids=list(range(length)), supervised=[True] * length,
        repository="r", path="p", source=source, tags={},
        tags_masked=False, tags_reversed=False,
    )


def groups(count, length, source, per_group=1):
    return [[segment(length, source) for _ in range(per_group)] for _ in range(count)]


def test_mix_holds_the_synthetic_share():
    real = groups(400, 1000, "gutenberg_fiction")
    synthetic = groups(400, 1000, "synthetic")
    rng = random.Random(0)
    chosen, tokens, synthetic_tokens = prepare.mix(real, synthetic, 0.05, None, rng)
    assert 0.03 <= synthetic_tokens / tokens <= 0.07, synthetic_tokens / tokens


def test_mix_respects_a_token_budget():
    real = groups(500, 1000, "gutenberg_fiction")
    synthetic = groups(100, 1000, "synthetic")
    rng = random.Random(1)
    _, tokens, _ = prepare.mix(real, synthetic, 0.05, 50_000, rng)
    # It stops at the first group that reaches the budget, so it may overshoot
    # by at most one group.
    assert 50_000 <= tokens <= 51_000, tokens


def test_mix_keeps_each_document_whole_and_in_order():
    # Consecutive windows of a book are packed contiguously and marked as
    # continuations. Splitting or reordering a group would make that a lie.
    real = groups(50, 100, "gutenberg_fiction", per_group=4)
    rng = random.Random(2)
    chosen, _, _ = prepare.mix(real, [], 0.0, None, rng)
    for group in chosen:
        assert len(group) == 4
        assert all(s.source == "gutenberg_fiction" for s in group)
    assert len(chosen) == 50


def test_mix_without_synthetic_is_all_real():
    rng = random.Random(3)
    chosen, tokens, synthetic_tokens = prepare.mix(
        groups(20, 500, "gutenberg_fiction"), [], 0.35, None, rng
    )
    assert synthetic_tokens == 0 and tokens == 20 * 500


def test_anneal_share_is_higher_than_the_pretraining_share():
    real = groups(600, 1000, "gutenberg_fiction")
    synthetic = groups(600, 1000, "synthetic")
    rng = random.Random(4)
    _, main_tokens, main_syn = prepare.mix(real, synthetic, 0.05, 200_000, rng)
    _, anneal_tokens, anneal_syn = prepare.mix(real, synthetic, 0.35, 200_000, rng)
    assert anneal_syn / anneal_tokens > main_syn / main_tokens * 3
