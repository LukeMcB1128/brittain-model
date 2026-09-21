"""Replace the mangled regex lines. Written as a file so no shell layer eats
the backslashes -- \b is a valid Python escape (backspace), which is exactly how
they got eaten the first time."""
import pathlib

p = pathlib.Path(r"C:\Coding\brittain-model\src\brittain\story_tagger.py")
lines = p.read_text(encoding="utf-8").split("\n")

replacements = {
    748: r'            rf"\b{escaped}\b[^.!?;]{{0,60}}?\b(himself|herself)\b",',
    750: r'            rf"\b{escaped}\s+(?:{_SPEECH_VERB})\b[^.!?;]{{0,15}}?,\s+(his|her)\b",',
    752: r'            rf"\b(his|her)\b[^.!?;]{{0,30}}?,?\s+(?:{_SPEECH_VERB})\s+{escaped}\b",',
    754: r'            rf"\b{escaped}\s+(?:is|was)\s+(?:an?\s+)?(?:\w+\s+){{0,2}}?({_DESCRIBED_AS})\b",',
}
for number, text in replacements.items():
    index = number - 1
    assert "escaped" in lines[index] or "his|her" in lines[index], lines[index]
    lines[index] = text

# Any stray backspace anywhere in the file is from the same accident.
body = "\n".join(lines)
assert "\x08" not in body, "backspace still present"
p.write_text(body, encoding="utf-8")
print("patterns repaired")
