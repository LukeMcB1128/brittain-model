"""
Interactive inference for BRITTAIN v2. Streams completions token-by-token.

    python3 sample.py

No context migration hacks needed — RoPE means the model just works at its
trained context length (and degrades gracefully a bit beyond it).
"""
import pickle

import torch
import tiktoken

from model import Brittain, GPTConfig

import sys
CKPT = sys.argv[1] if len(sys.argv) > 1 else "brittain_mac.pt"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

enc = tiktoken.get_encoding("gpt2")
ck = torch.load(CKPT, map_location=device)
cfg = GPTConfig(**ck['cfg'])
model = Brittain(cfg).to(device)
model.load_state_dict(ck['model'])
model.eval()
print(f"Loaded {CKPT} ({model.num_params():,} params) at iter {ck.get('iter', '?')}")
print("-" * 60)

while True:
    try:
        prompt = input("\nPrompt: ")
        if not prompt:
            continue
        ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
        print(prompt, end="", flush=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for _ in range(400):
                out = model.generate(ids, max_new_tokens=1, temperature=0.8, top_k=200)
                nxt = out[0, -1].item()
                ids = out
                print(enc.decode([nxt]), end="", flush=True)
        print("\n" + "-" * 40)
    except KeyboardInterrupt:
        print("\nbye")
        break
