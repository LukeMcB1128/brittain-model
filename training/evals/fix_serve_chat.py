"""Teach serve.py the story-chat frame. Written as a file so the shell does not
eat the pipe characters in the special tokens."""
import pathlib

p = pathlib.Path(r"C:\Coding\brittain-model\scripts\inference\serve.py")
s = p.read_text(encoding="utf-8")

# 1. Detect a prose model, and stop only on the tokens that actually end a turn.
old = '''        self.card = load_card(path, ck if isinstance(ck, dict) else {})'''
new = '''        specials = getattr(self.enc, "special_ids", {}) or {}
        # A prose model is one whose vocabulary has story markers. Asked of the
        # tokenizer rather than the filename, which says nothing reliable.
        self.story = "<|story_start|>" in specials
        if self.story:
            # Stopping on every special token is right for the code models,
            # whose specials all end something. It is wrong here: an SFT
            # response OPENS with <|tags|>, so the generation would stop on its
            # first token and return an empty story. Only these end a turn.
            self.stop_ids = {
                self.enc.eot,
                *(specials[name] for name in
                  ("<|end_message|>", "<|story_end|>", "<|pad|>")
                  if specials.get(name) is not None),
            }
        else:
            self.stop_ids = {self.enc.eot, *specials.values()}
        self.card = load_card(path, ck if isinstance(ck, dict) else {})'''
assert s.count(old) == 1
s = s.replace(old, new)

# 2. Decide instruct-vs-raw from more than the filename. A checkpoint written to
#    a directory is always "weights.pt", so the name carries nothing.
old = '''        low = os.path.basename(path).lower()'''
new = '''        # The filename alone is not enough: a Brittain3 run writes weights.pt
        # inside a named directory, so the basename is the same for every model
        # and "sft" never appears in it. The display name the operator chose and
        # the output_dir the checkpoint was trained into both carry the intent.
        trained_into = ""
        if isinstance(ck, dict):
            trained_into = str((ck.get("training_config") or {}).get("output_dir") or "")
        low = " ".join((os.path.dirname(path), os.path.basename(path),
                        name, trained_into)).lower()'''
assert s.count(old) == 1
s = s.replace(old, new)

# 3. The frame itself.
old = '''def stream_pieces(M, prompt, raw, opts):'''
new = '''def frame_request(M, text):
    """Wrap one request the way the model was fine-tuned to receive it.

    The Alpaca template is right for the code models and wrong for
    brittain-shakespeare, which learned <|user|>...<|end_message|><|assistant|>
    and answers with a tag block followed by the story.
    """
    if M.story:
        return f"<|user|>{text}<|end_message|><|assistant|>"
    return format_prompt(text)


def stream_pieces(M, prompt, raw, opts):'''
assert s.count(old) == 1
s = s.replace(old, new)

s = s.replace('''            stop_ids = {M.enc.eot, *getattr(M.enc, "special_ids", {}).values()}
            if nxt in stop_ids:''',
'''            if nxt in M.stop_ids:''')

# 4. Both request paths use it.
assert s.count("        prompt = format_prompt(prompt)") == 1
s = s.replace("        prompt = format_prompt(prompt)",
              "        prompt = frame_request(M, prompt)")
assert s.count("        prompt = format_prompt(user)") == 1
s = s.replace("        prompt = format_prompt(user)",
              "        prompt = frame_request(M, user)")

p.write_text(s, encoding="utf-8")
print("patched")
