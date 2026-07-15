"""
Build instruction-tuning data from Alpaca (cleaned), with LOSS MASKING.

For each (instruction, response) pair we store two aligned arrays:
  input_ids : the full  <prompt><response><eot>  token sequence
  labels    : same tokens, but the PROMPT positions are set to -100 so the loss
              ignores them. The model reads the instruction (context) but is only
              graded on producing the response — that's what turns "continue text"
              into "answer the instruction". -100 is PyTorch cross-entropy's
              default ignore_index, so masked positions contribute no gradient.

Run on the box:  pip install datasets ; python3 prepare_sft.py
Writes data/sft_input_ids.npy (uint16) and data/sft_labels.npy (int32).
"""
import os

import numpy as np
import tiktoken
from datasets import load_dataset

from sft_prompt import format_prompt

MAX_LEN = 512          # Alpaca examples are short; longer ones are truncated
OUT = "./data"
os.makedirs(OUT, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

print("Loading yahma/alpaca-cleaned ...")
ds = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"  {len(ds)} raw examples")

input_ids, labels, skipped = [], [], 0
for ex in ds:
    prompt = format_prompt(ex["instruction"], ex["input"])
    p = enc.encode_ordinary(prompt)
    r = enc.encode_ordinary(ex["output"]) + [EOT]     # response, ending in <eot>
    if len(p) >= MAX_LEN - 1:                          # no room for a response
        skipped += 1
        continue
    ids = (p + r)[:MAX_LEN]
    lab = ([-100] * len(p) + r)[:MAX_LEN]              # mask the prompt
    pad = MAX_LEN - len(ids)
    ids += [EOT] * pad
    lab += [-100] * pad                                # padding isn't graded
    input_ids.append(ids)
    labels.append(lab)

input_ids = np.array(input_ids, dtype=np.uint16)
labels = np.array(labels, dtype=np.int32)              # int32: holds token ids AND -100
np.save(os.path.join(OUT, "sft_input_ids.npy"), input_ids)
np.save(os.path.join(OUT, "sft_labels.npy"), labels)
print(f"Wrote {len(input_ids)} examples (skipped {skipped} over-long), "
      f"shape {input_ids.shape} -> data/sft_*.npy")
