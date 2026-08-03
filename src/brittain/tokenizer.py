"""
One tokenizer interface for BOTH models, so BRITTAIN-1 and BRITTAIN-2 run from
the same codebase:

  * BRITTAIN-1  -> gpt2 BPE via tiktoken            (vocab 50257)
  * BRITTAIN-2  -> our code BPE via tokenizers
                   (vocab 32000, tokenizers/brittain2-code-32k/tokenizer.json)

Checkpoints record their tokenizer; older v1 checkpoints don't, so we fall back
to inferring it from vocab_size.

Both wrappers expose the same three things generation needs:
    .encode(text) -> list[int]
    .token_bytes(id) -> bytes      (raw bytes, fed to an incremental UTF-8
                                    decoder so multi-byte chars stream correctly)
    .eot                            (end-of-text id; stops generation)
"""
import functools
from pathlib import Path

from .paths import BASE_TOKENIZER, FIM_TOKENIZER, PROJECT_ROOT


@functools.lru_cache(maxsize=1)
def _byte_decoder():
    """Inverse of GPT-2's bytes<->unicode mapping, used to recover the raw bytes
    behind a byte-level BPE token string."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class GPT2Tok:
    name = "gpt2"

    # Part of the tokenizer interface, not a CodeTok extra. BRITTAIN-1 predates
    # fill-in-the-middle and will never have sentinels, but callers still have to
    # ask — sample.py's `if enc.has_fim` raised AttributeError on every gpt2
    # checkpoint, and serve.py only survived by reaching for
    # getattr(enc, "has_fim", False). Declaring it here means neither has to.
    has_fim = False
    fim_prefix = fim_suffix = fim_middle = None

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self.eot = self._enc.eot_token
        self.vocab_size = self._enc.n_vocab

    def encode(self, text):
        return self._enc.encode_ordinary(text)

    def token_bytes(self, i):
        return self._enc.decode_single_token_bytes(i)

    def decode(self, ids):
        return self._enc.decode(ids)


def _resolve_tokenizer_path(path):
    """Resolve current paths while remaining compatible with older checkpoints."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    if not candidate.is_absolute() and (PROJECT_ROOT / candidate).exists():
        return PROJECT_ROOT / candidate
    if candidate.name == "code_bpe_fim.json" and FIM_TOKENIZER.exists():
        return FIM_TOKENIZER
    if candidate.name in {"code_bpe.json", "tokenizer.json"} and BASE_TOKENIZER.exists():
        return BASE_TOKENIZER
    return candidate


class CodeTok:
    name = "code_bpe"

    def __init__(self, path=BASE_TOKENIZER):
        from tokenizers import Tokenizer
        path = _resolve_tokenizer_path(path)
        self._tok = Tokenizer.from_file(str(path))
        self.eot = self._tok.token_to_id("<|endoftext|>")
        self.vocab_size = self._tok.get_vocab_size()
        self._bd = _byte_decoder()
        self.path = path
        # present only on the FIM-extended tokenizer; None otherwise
        self.fim_prefix = self._tok.token_to_id("<fim_prefix>")
        self.fim_suffix = self._tok.token_to_id("<fim_suffix>")
        self.fim_middle = self._tok.token_to_id("<fim_middle>")

    @property
    def has_fim(self):
        return None not in (self.fim_prefix, self.fim_suffix, self.fim_middle)

    def encode(self, text):
        return self._tok.encode(text).ids

    def token_bytes(self, i):
        tokstr = self._tok.id_to_token(i)
        return bytes(self._bd[c] for c in tokstr)

    def decode(self, ids):
        return self._tok.decode(ids)


def load_tokenizer(ck, code_bpe_path=BASE_TOKENIZER):
    """Pick the right tokenizer for a loaded checkpoint dict.

    A FIM checkpoint records tokenizer_path pointing at tokenizer_fim.json, whose
    vocab is 3 larger (the sentinels). Loading it with the base tokenizer.json
    would mismatch the model's embedding, so the checkpoint's own path wins.
    """
    name = ck.get("tokenizer")
    if name is None:                                  # v1 checkpoints predate the field
        name = "gpt2" if ck["cfg"]["vocab_size"] > 40000 else "code_bpe"
    if name == "gpt2":
        enc = GPT2Tok()
    else:
        enc = CodeTok(ck.get("tokenizer_path") or code_bpe_path)
    want = ck.get("cfg", {}).get("vocab_size")
    if want is not None and enc.vocab_size != want:
        raise ValueError(
            f"tokenizer vocab {enc.vocab_size} != model vocab {want}. "
            f"This checkpoint needs {ck.get('tokenizer_path') or name!r}; "
            f"a FIM model cannot use the base tokenizer.json and vice versa.")
    return enc
