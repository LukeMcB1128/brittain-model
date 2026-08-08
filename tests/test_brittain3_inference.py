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
