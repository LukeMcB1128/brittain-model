"""
One tokenizer interface for BOTH models, so BRITTAIN-1 and BRITTAIN-2 run from
the same codebase:

  * BRITTAIN-1  -> gpt2 BPE via tiktoken            (vocab 50257)
  * BRITTAIN-2  -> our code BPE via tokenizers      (vocab 32000, data/code_bpe.json)

Checkpoints record their tokenizer; older v1 checkpoints don't, so we fall back
to inferring it from vocab_size.

Both wrappers expose the same three things generation needs:
    .encode(text) -> list[int]
    .token_bytes(id) -> bytes      (raw bytes, fed to an incremental UTF-8
                                    decoder so multi-byte chars stream correctly)
    .eot                            (end-of-text id; stops generation)
"""
import functools


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


class CodeTok:
    name = "code_bpe"

    def __init__(self, path="data/code_bpe.json"):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self.eot = self._tok.token_to_id("<|endoftext|>")
        self.vocab_size = self._tok.get_vocab_size()
        self._bd = _byte_decoder()

    def encode(self, text):
        return self._tok.encode(text).ids

    def token_bytes(self, i):
        tokstr = self._tok.id_to_token(i)
        return bytes(self._bd[c] for c in tokstr)

    def decode(self, ids):
        return self._tok.decode(ids)


def load_tokenizer(ck, code_bpe_path="data/code_bpe.json"):
    """Pick the right tokenizer for a loaded checkpoint dict."""
    name = ck.get("tokenizer")
    if name is None:                                  # v1 checkpoints predate the field
        name = "gpt2" if ck["cfg"]["vocab_size"] > 40000 else "code_bpe"
    return GPT2Tok() if name == "gpt2" else CodeTok(code_bpe_path)
