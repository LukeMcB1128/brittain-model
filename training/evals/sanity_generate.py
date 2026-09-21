"""Prove the text-only class really loaded the multimodal checkpoint's weights.

The checkpoint stores model.language_model.layers.N.*; Qwen3_5ForCausalLM
expects model.layers.N.*. If that remap silently missed, from_pretrained would
hand back randomly-initialised layers and every later measurement would be of
a model that cannot generate. Random-token loss cannot distinguish the two --
coherent text can.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-9B"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
    device_map={"": 0},
)
model.eval()
print("class:", type(model).__name__)

msgs = [{"role": "user", "content": "In one sentence, what does a compiler do?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=80, do_sample=False)
print("\n--- generation ---")
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False))

# A loaded model assigns far lower loss to its own text than to random tokens.
real = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").to("cuda")
with torch.no_grad():
    real_loss = model(**real, labels=real["input_ids"]).loss.item()
    rand = torch.randint(0, 248320, (1, 128), device="cuda")
    rand_loss = model(input_ids=rand, labels=rand).loss.item()
print("\nloss on real text  : %.3f" % real_loss)
print("loss on random ids : %.3f" % rand_loss)
print("verdict:", "WEIGHTS LOADED" if real_loss < 5 else "SUSPECT -- looks untrained")
