"""brittain-shakespeare prose tokenizer interface and validation.

A sibling of ``tokenizer_v3`` rather than a parameterization of it:
``tokenizer_v3.validate_tokenizer`` hard-asserts a vocabulary of 24,576 and
validates against code samples, neither of which applies here.

The chat tokens are reserved from the start even though instruction tuning comes
later. Adding a special token after pretraining strands its embedding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .paths import PROJECT_ROOT
from .tokenizer import _byte_decoder

STORY_TOKENIZER_DIR = PROJECT_ROOT / "tokenizers" / "brittain-shakespeare-prose-8k"
STORY_TOKENIZER = STORY_TOKENIZER_DIR / "tokenizer.json"

STORY_VOCAB_SIZE = 8192

STORY_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|pad|>",
    "<|story_start|>",
    "<|story_end|>",
    "<|tags|>",
    "<|end_tags|>",
    "<|title|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_message|>",
)


class StoryTokenizer:
    """Small common interface for the brittain-shakespeare prose BPE."""

    name = "brittain_shakespeare_bpe"

    def __init__(self, path: str | Path = STORY_TOKENIZER):
        from tokenizers import Tokenizer

        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and (PROJECT_ROOT / candidate).exists():
            candidate = PROJECT_ROOT / candidate
        self.path = candidate.resolve()
        self._tok = Tokenizer.from_file(str(self.path))
        self.vocab_size = self._tok.get_vocab_size()
        self._byte_decoder = _byte_decoder()
        self.special_ids = {
            token: self._tok.token_to_id(token) for token in STORY_SPECIAL_TOKENS
        }
        missing = [token for token, value in self.special_ids.items() if value is None]
        if missing:
            raise ValueError(f"prose tokenizer is missing special tokens: {missing}")
        self.eot = self.special_ids["<|endoftext|>"]
        self.pad = self.special_ids["<|pad|>"]
        self.story_start = self.special_ids["<|story_start|>"]
        self.story_end = self.special_ids["<|story_end|>"]
        self.tags_start = self.special_ids["<|tags|>"]
        self.tags_end = self.special_ids["<|end_tags|>"]

    @property
    def has_fim(self) -> bool:
        # Fill-in-the-middle is a code-completion affordance. A story model does
        # not use it, and the sentinels are deliberately absent from the vocab.
        return False

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_bytes(self, token_id: int) -> bytes:
        token = self._tok.id_to_token(token_id)
        if token is None:
            return b""
        if token in self.special_ids:
            return token.encode("utf-8")
        try:
            return bytes(self._byte_decoder[character] for character in token)
        except KeyError:
            return self.decode([token_id]).encode("utf-8", errors="replace")


VALIDATION_SAMPLES = {
    "modern": "The wooden door slammed, and the room went very quiet.\n",
    "early_modern": "Thou art come betimes, and thy face doth tell me what "
                    "thy tongue would hide. 'Tis o'er.\n",
    "dialogue": '"I am here," she said. "I have always been here."\n',
    "tags": "<|tags|>[Voice: Modern] [Genre: Tragedy] [Setting: Tavern]<|end_tags|>",
    "names": "Harker Marlow Bathsheba Fitzwilliam\n",
    "punctuation": "He paused—then, quietly: 'Well?' … and left.\n",
    "unicode": "naïve café — 東京 — 🧪\n",
}


def validate_tokenizer(tokenizer: StoryTokenizer, *, reference=None) -> dict:
    """Validate required behavior and return comparable efficiency metrics."""
    if tokenizer.vocab_size != STORY_VOCAB_SIZE:
        raise ValueError(
            f"prose tokenizer vocab must be {STORY_VOCAB_SIZE}, got {tokenizer.vocab_size}"
        )
    for token, token_id in tokenizer.special_ids.items():
        encoded = tokenizer.encode(token)
        if encoded != [token_id]:
            raise ValueError(f"special token is not atomic: {token!r} -> {encoded}")

    metrics = {}
    for name, text in VALIDATION_SAMPLES.items():
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise ValueError(f"byte round trip failed for {name}: {decoded!r} != {text!r}")
        row = {"bytes": len(text.encode("utf-8")), "tokens": len(ids)}
        row["bytes_per_token"] = row["bytes"] / row["tokens"] if row["tokens"] else 0.0
        if reference is not None:
            reference_ids = reference.encode(text)
            row["reference_tokens"] = len(reference_ids)
            row["reduction"] = (
                1.0 - len(ids) / len(reference_ids) if reference_ids else 0.0
            )
        metrics[name] = row
    return metrics
