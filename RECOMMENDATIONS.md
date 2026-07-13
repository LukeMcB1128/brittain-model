# BRITTAIN — rework recommendations

Context: 604M-param char-level MoE transformer, trained ~9h / 51k iters at
context 32 on an M3 Max (38 GB), then patched to context 256 via position-table
migration/tiling hacks. Goal: a proof-of-concept for light coding + chatting.

The honest headline: **the model is ~50x too big for the data, char-level
tokenization is wasting most of its capacity, and the attention/MoE code is slow
on MPS.** Fixing those three things gets you a better model that trains in a
fraction of the time. The reworked code (`model.py`, `prepare.py`, `train.py`,
`sample.py`) already implements the recommendations below.

---

## 1. Tokenizer — the single biggest win

**Change char-level (vocab 187) → byte-level BPE (gpt2, vocab 50257).** Already
done in the new `prepare.py`.

- Char-level makes every token one character. Your 256-token context was ~40
  words, and most of the network's job was learning to *spell*.
- BPE packs ~4 chars/token, so the same context length holds ~4x more real text
  and the model sees words / common code fragments as single units. Coherence
  per parameter goes up a lot.
- RoPE (below) + BPE also kills the "context cliff" that forced the migrate/tile
  scripts — those are no longer needed.

## 2. Parameter count — go SMALLER, not bigger

Chinchilla-optimal is ~20 tokens per parameter. Your corpus is ~11M characters ≈
**~3M BPE tokens**. That supports roughly a **10–30M-parameter** model well;
604M is starved of data by ~1000x and just memorizes.

Presets in the new `model.py` (params include the tied 50257-row embedding):

| Preset | n_embd | n_layer | n_head | ctx | Params | Use |
|---|---|---|---|---|---|---|
| **Fast** (default in train.py) | 512 | 8 | 8 | 512 | ~51M | iterate in ~1–2h |
| **Quality** | 768 | 12 | 12 | 1024 | ~124M | best on this Mac, overnight |

Note ~25M of the "Fast" preset is the vocab embedding itself. Non-embedding
compute is small, which is why it's fast. **Don't exceed ~150M until you have
much more data** — see §5.

## 3. Context length

RoPE (rotary embeddings) replaces the learned absolute position table, so:

- Pick a context and just train at it. **512 is a good default**; 1024 for the
  Quality preset if memory allows.
- No more migrate.py / migrate_tile.py / finetune_positions.py. RoPE also
  extrapolates a bit past the trained length instead of falling off a cliff.

## 4. Training speed (M3 Max)

Implemented in the new `train.py`:

- **bf16 autocast** — M3 Max supports it; big speedup, half the memory.
- **Fused attention** via `F.scaled_dot_product_attention` + a single fused QKV
  projection, replacing the Python `ModuleList`-of-heads loop (which was the
  main slowness in the old code).
- **Dense SwiGLU FFN** instead of the Python-loop top-1 MoE. At this scale MoE
  added 4x FFN memory and slow boolean indexing for little quality gain.
- **Gradient accumulation** to get a large effective batch without OOM.
- **Cosine LR + warmup + grad clipping** — stable at a higher learning rate.

Measured on your M3 Max: the Fast (51M) preset runs ~0.34 s/step at ctx256/bs8.
A full Fast run is on the order of an hour or two, versus the 9 hours the 604M
char model took.

**Biggest speed lever of all: rent a GPU.** A single RTX 4090 or A100 spot
instance (a few $/hr on Vast.ai / RunPod / Lambda) trains these models in
*minutes*, not hours. If you keep iterating, that's worth far more than any
local tuning. The code is portable — change `mps` to `cuda`.

## 5. Data — your real bottleneck

~3M tokens is not enough for coherent chat/code. To actually get a fun PoC:

- **Add public data.** For readable English + light code flavor, mix in
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (simple,
  coherent, ~500M tokens) and/or a small slice of a code dataset (e.g. The Stack
  smol, or just more of your own repos). Aim for **50–500M tokens**.
- Keep a held-out val set and watch the train/val gap in `train.py`'s eval — if
  val stops improving while train keeps dropping, you're overfitting (need more
  data or a smaller model).
- Only after you have 100M+ tokens does scaling params past ~150M make sense.

---

## What to run

```bash
pip install tiktoken                     # one-time (already installed)
python3 prepare.py                       # rebuild data as BPE
python3 train.py                         # trains the Fast preset -> brittain_v2.pt
python3 sample.py                        # chat with it
```

Old files (`transformer.py`, `generate.py`, `migrate*.py`,
`finetune_positions.py`) are kept in git history / the repo for reference but are
superseded by the four new scripts.
