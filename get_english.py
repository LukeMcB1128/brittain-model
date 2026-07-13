"""
Download a public-domain English corpus from Project Gutenberg into
data/english.txt. This is the English half of the training mix (see
RECOMMENDATIONS.md): a small code-only model can't produce coherent language,
so we pretrain-mix real prose alongside the code.

(TinyStories would give simpler English, but its HuggingFace CDN is not
reachable from every environment. Gutenberg is plain-text and reliable.)

Run once:  python3 get_english.py
"""
import os
import re
import time
import subprocess

OUT = "data/english.txt"
os.makedirs("data", exist_ok=True)

# Mix of simpler/narrative works and larger novels for volume.
BOOK_IDS = [
    11, 12, 16, 55, 74, 76, 236, 271, 289, 113,   # children's / narrative
    45, 120, 289, 2591, 1934, 19033, 15, 35, 36, 43,
    84, 98, 730, 1260, 768, 158, 1342, 161, 141, 145,   # 19th-c novels
    1400, 786, 174, 1661, 2701, 135, 1184, 2600, 100, 996,  # big volume
]

HEADER = re.compile(r"\*\*\* ?START OF (THE|THIS)? ?PROJECT GUTENBERG.*?\*\*\*",
                    re.IGNORECASE | re.DOTALL)
FOOTER = re.compile(r"\*\*\* ?END OF (THE|THIS)? ?PROJECT GUTENBERG.*",
                    re.IGNORECASE | re.DOTALL)

def fetch(book_id):
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ]
    for url in urls:
        try:
            out = subprocess.run(
                ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "60", url],
                capture_output=True, timeout=90,
            )
            raw = out.stdout.decode("utf-8", errors="ignore")
            if len(raw) > 1000 and "<!DOCTYPE" not in raw[:200]:
                return raw
        except Exception:
            continue
    return None

def strip_boilerplate(text):
    m = HEADER.search(text)
    if m:
        text = text[m.end():]
    m = FOOTER.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()

total = 0
seen = set()
with open(OUT, "w", encoding="utf-8") as out:
    for bid in BOOK_IDS:
        if bid in seen:
            continue
        seen.add(bid)
        raw = fetch(bid)
        if not raw:
            print(f"  [skip] {bid} (unreachable)")
            continue
        body = strip_boilerplate(raw)
        out.write(body + "\n\n")
        total += len(body)
        print(f"  [ok]   {bid:6d}  {len(body)/1e6:5.2f} MB   (running: {total/1e6:.1f} MB)")
        time.sleep(0.5)  # be polite to Gutenberg

print(f"\nWrote {OUT}: {total/1e6:.1f} MB  (~{total/4/1e6:.1f}M tokens)")
