import random
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.data_v3 import (
    Document,
    FIMSettings,
    encode_document,
    pack_segments,
    select_hole,
    split_by_repository,
    token_controlled_mix,
)
from brittain.tokenizer_v3 import BRITTAIN3_SPECIAL_TOKENS, Brittain3Tokenizer


@pytest.fixture()
def tokenizer(tmp_path):
    output = tmp_path / "tokenizer.json"
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=list(BRITTAIN3_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    samples = [
        "def parseConfig(value):\n    return value.strip()\n",
        "function renderApp(state) { return state; }\n",
        "A tool call uses JSON arguments and a stable result.\n",
    ] * 100
    tokenizer.train_from_iterator(samples, trainer=trainer)
    tokenizer.save(str(output))
    return Brittain3Tokenizer(output)


def documents():
    return [
        Document("repo-a", "src/a.py", "def alpha():\n    return 1\n", "python", True),
        Document("repo-a", "README.md", "Alpha project documentation.\n", "english", False),
        Document("repo-b", "src/b.py", "def beta():\n    return 2\n", "python", True),
        Document("repo-c", "src/c.js", "function gamma() { return 3; }\n", "javascript", True),
    ]


def test_repository_split_has_no_leakage():
    train, validation = split_by_repository(documents(), 0.5, 7)
    train_repos = {item.repository for item in train}
    validation_repos = {item.repository for item in validation}
    assert not train_repos & validation_repos
    assert len(train) + len(validation) == len(documents())


@pytest.mark.parametrize("psm_rate, expected", [(1.0, "PSM"), (0.0, "SPM")])
def test_fim_orders_are_complete_and_indivisible(tokenizer, psm_rate, expected):
    document = Document(
        "repo", "src/sample.py",
        "def first():\n    value = 1\n    return value\n\ndef second():\n    return 2\n",
        "python", True,
    )
    settings = FIMSettings(rate=1.0, psm_rate=psm_rate, line_rate=1.0, block_rate=0.0)
    segment = encode_document(document, tokenizer, 128, settings, random.Random(4))
    assert segment.is_fim and segment.fim_order == expected
    for token in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>"):
        assert tokenizer.special_ids[token] in segment.ids
    inputs, labels, spans = pack_segments([segment], 128, tokenizer.pad)
    assert inputs.shape == labels.shape == (1, 128)
    assert spans[0][0]["start"] == 0
    assert spans[0][0]["end"] == len(segment.ids)
    assert spans[0][0]["is_fim"] is True


def test_fim_is_only_applied_to_code(tokenizer):
    prose = Document("repo", "README.md", "One line.\nTwo lines.\nThree lines.\n", "english", False)
    segment = encode_document(
        prose, tokenizer, 128, FIMSettings(rate=1.0), random.Random(1)
    )
    assert not segment.is_fim
    assert tokenizer.fim_prefix not in segment.ids


@pytest.mark.parametrize(
    "settings, expected",
    [
        (FIMSettings(rate=1.0, line_rate=1.0, block_rate=0.0), "line"),
        (FIMSettings(rate=1.0, line_rate=0.0, block_rate=1.0), "block"),
        (FIMSettings(rate=1.0, line_rate=0.0, block_rate=0.0), "random"),
    ],
)
def test_all_fim_hole_kinds(settings, expected):
    text = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    selected = select_hole(text, settings, random.Random(3))
    assert selected is not None
    prefix, middle, suffix, kind = selected
    assert kind == expected
    assert prefix + middle + suffix == text


def test_block_hole_keeps_one_javascript_function():
    text = (
        "function first() {\n  return 1;\n}\n\n"
        "const second = () => {\n  return 2;\n};\n"
    )
    selected = select_hole(
        text, FIMSettings(rate=1.0, line_rate=0.0, block_rate=1.0), random.Random(4)
    )
    assert selected is not None
    prefix, middle, suffix, kind = selected
    assert kind == "block"
    assert prefix + middle + suffix == text
    assert not ("first" in middle and "second" in middle)


def test_packer_never_splits_fim_across_rows(tokenizer):
    settings = FIMSettings(rate=1.0, psm_rate=1.0, line_rate=1.0, block_rate=0.0)
    segments = [
        encode_document(
            Document(f"repo-{i}", f"src/{i}.py", "def f():\n    x = 1\n    return x\n", "python", True),
            tokenizer, 96, settings, random.Random(i),
        ) for i in range(6)
    ]
    _, _, rows = pack_segments(segments, 96, tokenizer.pad)
    assert sum(len(row) for row in rows) == len(segments)
    assert all(span["is_fim"] for row in rows for span in row)
    for row in rows:
        assert all(span["start"] < span["end"] <= 97 for span in row)


def test_token_controlled_mix_is_deterministic_and_uses_all_documents():
    docs = documents()
    groups = {
        "python": [item for item in docs if item.source == "python"],
        "english": [item for item in docs if item.source == "english"],
        "javascript": [item for item in docs if item.source == "javascript"],
    }
    weights = {"python": 0.6, "english": 0.2, "javascript": 0.2}
    first = token_controlled_mix(groups, weights, lambda item: len(item.text), 9)
    second = token_controlled_mix(groups, weights, lambda item: len(item.text), 9)
    assert first == second
    assert set(first) == set(docs)


def test_oversized_metadata_fails_clearly(tokenizer):
    document = Document("r" * 500, "p" * 500, "short", "python", True)
    with pytest.raises(ValueError, match="metadata"):
        encode_document(document, tokenizer, 32, FIMSettings(), random.Random(1))
