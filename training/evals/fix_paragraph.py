"""Repair the two lines whose newline escapes were eaten by the shell."""
import pathlib

p = pathlib.Path(r"C:\Coding\brittain-model\src\brittain\chat_context.py")
lines = p.read_text(encoding="utf-8").split("\n")

start = next(i for i, l in enumerate(lines) if "Whole paragraphs" in l)
# The broken region runs from the partition call to the break.
end = next(i for i, l in enumerate(lines[start:], start) if l.strip() == "break")
replacement = [
    "            # Whole paragraphs, so the continuation never opens mid-sentence.",
    '            _, _, body = body.partition("\\n\\n")',
    '            if "\\n\\n" not in body and len(encode(body)) > room:',
    "                body = body[-room * 4:]",
    "                break",
]
lines[start:end + 1] = replacement
body = "\n".join(lines)
assert "\x08" not in body
p.write_text(body, encoding="utf-8")
print("repaired")
