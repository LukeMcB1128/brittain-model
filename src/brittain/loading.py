"""One loader for every BRITTAIN checkpoint family.

`compare.py`, `novice.py`, and `sample.py` each grew their own copy of the
"which architecture is this?" branch. They drifted: `compare.py` never learned
to read Brittain3 at all, so the pilot could not be scored against the Brittain2
XS baselines it exists to beat. This module is the single branch.

Three families, distinguished by what `torch.load` returns:

- **Brittain3** — dict with `cfg` and `architecture == "brittain3"`.
- **Brittain1/2** — dict with `cfg` and no `architecture` key.
- **BrittainScript 50M** — a bare ModuleList state_dict written by
  `train_50m.bs`, with no `cfg` key at all.

`load_any` returns `(model, block_size, tokenizer)` so a caller can score all
three without caring which is which.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import model_bs
from .model import Brittain, GPTConfig
from .model_v3 import Brittain3, Brittain3Config
from .paths import BRITTAIN3_TOKENIZER
from .tokenizer import load_tokenizer


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tokenizer_for(checkpoint: dict) -> object:
    """Load the checkpoint's tokenizer, tolerating a stale absolute path.

    Checkpoints record `tokenizer_path` as an ABSOLUTE path from the machine
    that trained them, so a checkpoint moved off the training box raises
    FileNotFoundError on a path that means nothing here. The vocab check inside
    `load_tokenizer` still runs against the fallback, so a genuinely mismatched
    tokenizer is still rejected loudly rather than silently producing garbage.
    """
    recorded = checkpoint.get("tokenizer_path")
    if recorded and not Path(recorded).exists():
        checkpoint = dict(checkpoint)
        if checkpoint.get("tokenizer") == "brittain3_bpe":
            checkpoint["tokenizer_path"] = str(BRITTAIN3_TOKENIZER)
        else:
            checkpoint["tokenizer_path"] = None
    return load_tokenizer(checkpoint)


def load_any(path: str | Path, device: torch.device | str = "cpu"):
    """Return (model, block_size, tokenizer) for any BRITTAIN checkpoint."""
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "cfg" not in checkpoint:
        del checkpoint
        model, tokenizer = model_bs.load(str(path), device)
        return model, model.block, tokenizer

    if checkpoint.get("architecture") == "brittain3":
        config = Brittain3Config(**checkpoint["cfg"])
        model = Brittain3(config).to(device)
        block_size = config.max_seq_len
    else:
        config = GPTConfig(**checkpoint["cfg"])
        model = Brittain(config).to(device)
        block_size = config.block_size

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, block_size, _tokenizer_for(checkpoint)


def generate(model, ids: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
    """Generate from any family with one call signature.

    Brittain and Brittain3 take `max_new_tokens` as a keyword; the BrittainScript
    model takes it positionally and rejects the keyword.
    """
    if isinstance(model, (Brittain, Brittain3)):
        return model.generate(ids, max_new_tokens=max_new_tokens, **kwargs)
    return model.generate(ids, max_new_tokens, **kwargs)
