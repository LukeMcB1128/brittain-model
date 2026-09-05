import subprocess
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.model_v3 import Brittain3, Brittain3Config
from brittain.tokenizer import load_tokenizer
from brittain.tokenizer_v3 import BRITTAIN3_SPECIAL_TOKENS


def tokenizer_file(path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=list(BRITTAIN3_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        ["def add(a, b):\n    return a + b\n", "A short English sentence.\n"] * 100,
        trainer=trainer,
    )
    tokenizer.save(str(path))
    return tokenizer.get_vocab_size()


def test_shared_tokenizer_loader_and_sample_dispatch(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_size = tokenizer_file(tokenizer_path)
    cfg = Brittain3Config(
        vocab_size=vocab_size, max_seq_len=64, n_layer=1, n_head=2,
        n_kv_head=1, n_embd=16, intermediate_size=32,
        activation_checkpointing=False,
    )
    model = Brittain3(cfg)
    checkpoint = {
        "architecture": "brittain3", "architecture_version": 1,
        "cfg": cfg.to_dict(), "model": model.state_dict(),
        "tokenizer": "brittain3_bpe", "tokenizer_path": str(tokenizer_path),
        "iter": 0, "val": 1.0,
    }
    loaded = load_tokenizer(checkpoint)
    assert loaded.name == "brittain3_bpe"
    assert loaded.vocab_size == vocab_size
    checkpoint_path = tmp_path / "tiny.pt"
    torch.save(checkpoint, checkpoint_path)
    result = subprocess.run(
        [
            sys.executable, "scripts/inference/sample.py", str(checkpoint_path),
            "--prompt", "def add(a, b):\n", "--max_tokens", "2", "--top_k", "1",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "brittain3_bpe" in result.stdout
    assert "ctx 64" in result.stdout


# --------------------------------------------------------------------------- #
# Story sampler prompt framing
# --------------------------------------------------------------------------- #

def _story_module():
    import importlib.util
    path = PROJECT_ROOT / "scripts" / "inference" / "story.py"
    spec = importlib.util.spec_from_file_location("story_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    """Ids are distinct and small so the assertions read as sequences."""
    story_start, story_end, eot, tags_start, tags_end, pad = 1, 2, 3, 4, 5, 0
    special_ids = {"<|story_start|>": 1}

    def encode(self, text):
        return [10 + (ord(c) % 40) for c in text]


def test_a_continuation_does_not_cross_a_document_boundary():
    """The bug this guards against produced a stage play.

    Ending the previous text with story_end + eot is a document boundary, and
    the model read it as one: continuing a bar-room comedy produced "SCENE
    XXII" out of the corpus-wide document prior. A continuation has to extend
    the document that is already open.
    """
    story = _story_module()
    tokenizer = _FakeTokenizer()
    ids = story.build_prompt(tokenizer, {"Genre": "Comedy"}, "", previous="a story",
                             block=512)
    assert tokenizer.story_end not in ids
    assert tokenizer.eot not in ids
    assert ids[0] == tokenizer.story_start
    # The tag block leads, then the text so far, exactly as one long document.
    assert ids[1] == tokenizer.tags_start
    assert ids.index(tokenizer.tags_end) < len(ids) - 1


def test_a_continuation_keeps_the_most_recent_context_within_the_block():
    story = _story_module()
    tokenizer = _FakeTokenizer()
    previous = "x" * 4000
    ids = story.build_prompt(tokenizer, {}, "", previous=previous, block=1024)
    assert len(ids) <= 1024
    # Room is reserved for generation rather than filling the window.
    assert len(ids) <= 1024 - 256


def test_a_fresh_story_still_opens_with_the_start_marker():
    story = _story_module()
    tokenizer = _FakeTokenizer()
    ids = story.build_prompt(tokenizer, {}, "The door", block=512)
    assert ids[0] == tokenizer.story_start
