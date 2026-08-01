"""
Talk to the instruction-tuned BRITTAIN. Wraps your input in the SAME template
the model was fine-tuned on, then generates the response and stops at <|endoftext|>.

    python3 scripts/inference/chat.py
    python3 scripts/inference/chat.py checkpoints/brittain_124m_sft.pt

This is where you feel the SFT payoff: you give an instruction, it attempts an
answer, instead of just continuing your text.
"""
import sys
import codecs
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from brittain.model import Brittain, GPTConfig
from brittain.paths import CHECKPOINT_DIR
from brittain.prompts import format_prompt
from brittain.tokenizer import load_tokenizer

CKPT = (sys.argv[1] if len(sys.argv) > 1
        else str(CHECKPOINT_DIR / "brittain_124m_sft.pt"))
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
            # stream() holds the KV cache open for the whole response
            for tok in model.stream(ids, 400, temperature=0.7, top_p=0.9,
                                    repetition_penalty=1.3):
                nxt = tok[0, -1].item()
                if nxt == enc.eot:
                    break
                print(utf8.decode(enc.token_bytes(nxt)), end="", flush=True)
        print()
    except KeyboardInterrupt:
        print("\nbye")
        break
