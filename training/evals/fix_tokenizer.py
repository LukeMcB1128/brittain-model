"""Two serving fixes, applied as a file so no shell layer eats the backslashes."""
import pathlib

p = pathlib.Path(r"C:\Coding\brittain-model\src\brittain\tokenizer.py")
s = p.read_text(encoding="utf-8")

old_resolver = '''    if candidate.name == "code_bpe_fim.json" and FIM_TOKENIZER.exists():
        return FIM_TOKENIZER
    if candidate.name in {"code_bpe.json", "tokenizer.json"} and BASE_TOKENIZER.exists():
        return BASE_TOKENIZER
    return candidate'''
new_resolver = '''    # A checkpoint records the absolute path of the machine that trained it, so
    # one trained on a Mac names /Users/... and cannot be served on Windows. The
    # part from "tokenizers/" onward is the same in every clone, so re-root it.
    parts = candidate.parts
    if "tokenizers" in parts:
        tail = Path(*parts[parts.index("tokenizers"):])
        if (PROJECT_ROOT / tail).exists():
            return PROJECT_ROOT / tail
    if candidate.name in {"code_bpe_fim.json", "tokenizer_fim.json"} and FIM_TOKENIZER.exists():
        return FIM_TOKENIZER
    if candidate.name in {"code_bpe.json", "tokenizer.json"} and BASE_TOKENIZER.exists():
        return BASE_TOKENIZER
    return candidate'''
assert s.count(old_resolver) == 1
s = s.replace(old_resolver, new_resolver)

old_load = '''    if name == "gpt2":
        enc = GPT2Tok()
    elif name == "brittain3_bpe":
        from .tokenizer_v3 import Brittain3Tokenizer
        tokenizer_path = ck.get("tokenizer_path")
        enc = Brittain3Tokenizer(tokenizer_path) if tokenizer_path else Brittain3Tokenizer()'''
new_load = '''    if name == "gpt2":
        enc = GPT2Tok()
    elif name == "brittain3_bpe":
        tokenizer_path = ck.get("tokenizer_path")
        # "brittain3_bpe" names the architecture family, not the vocabulary.
        # brittain-shakespeare is a Brittain3 model on an 8K prose vocabulary
        # whose special tokens are story and chat markers, so validating it
        # against the code set rejects it for missing <|fim_prefix|>. The path
        # is what actually says which vocabulary this is.
        if tokenizer_path and _is_prose_tokenizer(tokenizer_path):
            from .tokenizer_story import StoryTokenizer
            enc = StoryTokenizer(tokenizer_path)
        else:
            from .tokenizer_v3 import Brittain3Tokenizer
            enc = Brittain3Tokenizer(tokenizer_path) if tokenizer_path else Brittain3Tokenizer()'''
assert s.count(old_load) == 1
s = s.replace(old_load, new_load)

helper = '''def _is_prose_tokenizer(path) -> bool:
    """Whether a tokenizer file carries the story vocabulary's markers.

    Asked of the file rather than of the checkpoint's label, because the label
    records the architecture and two different vocabularies share it.
    """
    from tokenizers import Tokenizer

    resolved = _resolve_tokenizer_path(path)
    if not Path(resolved).exists():
        return False
    try:
        return Tokenizer.from_file(str(resolved)).token_to_id("<|story_start|>") is not None
    except Exception:
        return False


def load_tokenizer(ck, code_bpe_path=BASE_TOKENIZER):'''
s = s.replace("def load_tokenizer(ck, code_bpe_path=BASE_TOKENIZER):", helper)

p.write_text(s, encoding="utf-8")
print("patched")
