import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.tokenizer_v3 import BRITTAIN3_SPECIAL_TOKENS, Brittain3Tokenizer


def make_tokenizer(path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1024,
        special_tokens=list(BRITTAIN3_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    corpus = [
        "def fibonacci(n: int):\n    return n\n",
        '{"name":"find_symbol","arguments":{"name":"parseConfig"}}',
        "A coding assistant reads the project documentation.\n",
        "naïve café — 東京 — 🧪\n",
    ] * 200
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save(str(path))


def test_special_tokens_are_atomic_and_text_round_trips(tmp_path):
    path = tmp_path / "tokenizer.json"
    make_tokenizer(path)
    tokenizer = Brittain3Tokenizer(path)
    for token, token_id in tokenizer.special_ids.items():
        assert tokenizer.encode(token) == [token_id]
    for text in (
        "def f():\n    return 1\n",
        "naïve café — 東京 — 🧪\n",
        '{"name":"read_file","arguments":{"path":"src/app.js"}}',
    ):
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids, skip_special_tokens=False) == text
        assert b"".join(tokenizer.token_bytes(token_id) for token_id in ids).decode() == text


def test_tool_call_structure_round_trips(tmp_path):
    path = tmp_path / "tokenizer.json"
    make_tokenizer(path)
    tokenizer = Brittain3Tokenizer(path)
    text = '<|assistant|><|tool_call|>{"name":"find_symbol","arguments":{"name":"parseConfig"}}<|end_message|>'
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids, skip_special_tokens=False) == text
    assert ids[0] == tokenizer.special_ids["<|assistant|>"]
    assert ids[1] == tokenizer.special_ids["<|tool_call|>"]
    assert ids[-1] == tokenizer.special_ids["<|end_message|>"]

