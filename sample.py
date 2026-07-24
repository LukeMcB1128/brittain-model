"""
Interactive inference for BRITTAIN v2. Streams completions token-by-token.

    python3 sample.py

No context migration hacks needed — RoPE means the model just works at its
trained context length (and degrades gracefully a bit beyond it).
"""
import pickle
import codecs

import torch

from model import Brittain, GPTConfig
from tok_util import load_tokenizer

import sys
CKPT = sys.argv[1] if len(sys.argv) > 1 else "brittain_124m_best.pt"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ck = torch.load(CKPT, map_location=device)
cfg = GPTConfig(**ck['cfg'])
model = Brittain(cfg).to(device)
model.load_state_dict(ck['model'])
model.eval()
enc = load_tokenizer(ck)   # gpt2 for v1 ckpts, code BPE for v2
print(f"Loaded {CKPT} ({model.num_params():,} params) at iter {ck.get('iter', '?')}")
print("-" * 60)

while True:
    try:
        prompt = input("\nPrompt: ")
        if not prompt:
            continue
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        print(prompt, end="", flush=True)
        # incremental UTF-8 decoder buffers multi-byte chars across tokens (no ��)
        utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for _ in range(400):
                ids = model.generate(ids, max_new_tokens=1, temperature=0.9,
                                     top_p=0.9, repetition_penalty=1.3)
                nxt = ids[0, -1].item()
                if nxt == enc.eot:          # stop at document boundary
                    break
                print(utf8.decode(enc.token_bytes(nxt)), end="", flush=True)
        print("\n" + "-" * 40)
    except KeyboardInterrupt:
        print("\nbye")
        break
