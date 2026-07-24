"""
Talk to the instruction-tuned BRITTAIN. Wraps your input in the SAME template
the model was fine-tuned on, then generates the response and stops at <|endoftext|>.

    python3 chat.py                      # uses brittain_124m_sft.pt
    python3 chat.py brittain_124m_sft.pt

This is where you feel the SFT payoff: you give an instruction, it attempts an
answer, instead of just continuing your text.
"""
import sys
import codecs

import torch

from model import Brittain, GPTConfig
from tok_util import load_tokenizer
from sft_prompt import format_prompt

CKPT = sys.argv[1] if len(sys.argv) > 1 else "brittain_124m_sft.pt"
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

ck = torch.load(CKPT, map_location=device)
cfg = GPTConfig(**ck['cfg'])
model = Brittain(cfg).to(device)
model.load_state_dict(ck['model'])
model.eval()
enc = load_tokenizer(ck)   # gpt2 for v1 ckpts, code BPE for v2
print(f"Loaded {CKPT} ({model.num_params():,} params). Ctrl-C to quit.")
print("-" * 60)

while True:
    try:
        instruction = input("\nInstruction: ")
        if not instruction.strip():
            continue
        prompt = format_prompt(instruction)
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        print("Response: ", end="", flush=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for _ in range(400):
                ids = model.generate(ids, max_new_tokens=1, temperature=0.7,
                                     top_p=0.9, repetition_penalty=1.3)
                nxt = ids[0, -1].item()
                if nxt == enc.eot:
                    break
                print(utf8.decode(enc.token_bytes(nxt)), end="", flush=True)
        print()
    except KeyboardInterrupt:
        print("\nbye")
        break
