# The BRITTAIN models

Every model here is a **decoder-only transformer** trained from scratch — no
pretrained weights, no fine-tuning of someone else's model. This document explains
the shared architecture, how it differs from GPT-2, and what each checkpoint is.

---

## The reference point: GPT-2 (OpenAI, 2019)

GPT-2 is the architecture every model here descends from. Its shape:

| | |
|---|---|
| Type | decoder-only transformer (causal LM) |
| Positions | **learned absolute** embeddings — a lookup table of `block_size` vectors |
| Attention | multi-head causal self-attention |
| MLP | `Linear(d → 4d)` → GELU → `Linear(4d → d)` |
| Norm | LayerNorm, pre-norm (at the input of each sub-block) + a final norm |
| Weight tying | token embedding shares weights with the output head |
| Tokenizer | byte-level BPE, vocab 50257 |
| Sizes | 124M / 355M / 774M / 1.5B |
| Training data | WebText — ~40GB of scraped, link-filtered web text |

**GPT-2 small** is 12 layers, 12 heads, 768 embedding dim, 1024 context = ~124M
parameters. That's the configuration BRITTAIN-1 deliberately matches.

The core idea hasn't changed since: predict the next token, stack N identical
blocks of (attention + MLP) with residual connections. Everything below is
refinements to *how* those pieces are implemented.

---

## What BRITTAIN changes vs GPT-2

Three substitutions, all standard practice post-2020 (they're what LLaMA-class
models use), implemented in `model.py`:

### 1. RoPE instead of learned position embeddings

GPT-2 learns a table of position vectors — one per slot, up to `block_size`. Two
problems: the table is fixed-size, and each slot must be *individually* trained.

**This was the bug that started this whole project.** The original model trained at
context 32, so only 32 position vectors were ever learned. Extending to 256 meant
224 untrained slots of random noise — output turned to gibberish past position 31.
The migrate/tile/finetune scripts in the git history are all failed attempts to
paper over that.

**RoPE (Rotary Position Embedding)** fixes it structurally: instead of *adding* a
learned position vector, it *rotates* the query and key vectors by an angle
proportional to position. Position information becomes relative and geometric
rather than learned per-slot — nothing to train, no table to outgrow, and it
degrades gracefully past the trained length instead of falling off a cliff.

### 2. SwiGLU instead of a GELU MLP

GPT-2's MLP is `Linear → GELU → Linear`. SwiGLU uses a *gated* form: two parallel
projections where one gates the other via SiLU, then a projection back down.
Better quality per parameter; standard since PaLM/LLaMA. Hidden size is set to
`8/3 × d` rather than `4 × d` to keep the parameter count comparable.

### 3. Fused attention

Instead of computing attention scores manually, the models call
`F.scaled_dot_product_attention`, which dispatches to a Flash-Attention-style
kernel — same math, dramatically less memory and much faster, because it never
materializes the full T×T attention matrix.

Also kept from GPT-2: pre-norm LayerNorm, weight tying, residual connections, and
a scaled init on residual projections.

---

## The models

### Prehistory — the 604M char-level MoE *(abandoned)*

The first attempt. 604M parameters, **character-level** tokenization (vocab 187),
a Mixture-of-Experts feed-forward with top-1 routing, learned positions, context
**32**. Trained ~9 hours on an M3 Max.

Why it was abandoned:
- **Char-level tokenization** meant each token was one character, so most of the
  network's capacity went into learning to *spell*, and a 32-token context was
  about six words.
- **604M parameters on ~3M tokens** of training data is roughly 1000× past the
  point of diminishing returns — it memorized rather than generalized.
- The context cliff described above made extending it impossible.

Preserved in git history at the `baseline` commit. Not runnable with current code.

### BRITTAIN-1 — 124M *(complete)*

| | |
|---|---|
| Parameters | 123,551,232 |
| Shape | 12 layers, 12 heads, 768 embd, 1024 context |
| Tokenizer | GPT-2 BPE, vocab 50257 |
| Data | FineWeb-Edu, ~2.6B tokens |
| Hardware | 1× NVIDIA L4, ~20 hours, ~$17 |
| Final val loss | **3.247** (perplexity ~26) |
| Post-training | SFT on Alpaca-cleaned, loss ~1.7 |
| Checkpoints | `brittain_124m_best.pt`, `brittain_124m_sft.pt` |

Deliberately GPT-2-small-shaped, so the result is comparable to a known reference.
Writes fluent, grammatical English. Confidently wrong on facts, no arithmetic, no
real reasoning — the expected ceiling at this scale.

The SFT stage (`train_sft.py`) is the same thing that turns a base model into a
chat model: train on instruction→response pairs wrapped in a fixed template, with
the **loss masked over the prompt** so the model learns to *answer* instructions
rather than generate them.

### BRITTAIN-2 — 235M coder *(in progress)*

| | |
|---|---|
| Parameters | 235,176,960 |
| Shape | 16 layers, 16 heads, 1024 embd, 1024 context |
| Tokenizer | **custom 32k code BPE** (`data/code_bpe.json`) |
| Data | The Stack (Python/JS/TS) + 15% FineWeb-Edu, ~14.7B tokens |
| Hardware | 1× NVIDIA L4, ~7.4 days, ~$135 |
| Expected val | ~1.30–1.40 |

Two deliberate departures from v1:

**A custom tokenizer.** GPT-2's BPE learned its merges on prose, where long runs of
indentation are rare — so it has no token for `"\n        "` and burns several
tokens per indented line. Retraining the merges on code yields **44% fewer tokens**
on a sample Python function. That's ~44% more code per training step *and* per
context window, for free. The smaller vocab (32k vs 50257) also frees ~14M
parameters from the embedding table to spend on actual layers.

**Deliberate over-training.** Chinchilla-optimal for 235M would be ~4.7B tokens;
this run does ~3× that. Chinchilla minimizes loss for a fixed compute budget when
you're free to pick the model size — but the goal here is the *smallest model that's
actually good*, so you fix the size and keep feeding it data. Standard modern
practice (LLaMA-3 took this to ~90× Chinchilla).

### brittain2-xs-coder:50m-bs — 52M *(proof of concept)*

| | |
|---|---|
| Parameters | 51,917,824 |
| Shape | 6 layers, 8 heads, 512 embd, 512 context |
| Tokenizer | same 32k code BPE as v2 |
| Data | same corpus as v2, ~1B tokens |
| Hardware | M3 Max (MPS), ~15 hours |
| Trained by | **`train_50m.bs` — BrittainScript** |

Same data and tokenizer as BRITTAIN-2, but **the training script is written in
[BrittainScript](https://pypi.org/project/brittainscript/)**, a language with its
own interpreter, driving PyTorch through the Python bridge added in v0.3.0.

The architecture is simplified because BrittainScript can't express the v2 feature
set:

| v2 uses | 50m-bs uses | why |
|---|---|---|
| RoPE | learned positions | RoPE needs cos/sin tables and tensor chunking that's painful without n-d indexing |
| SwiGLU | GELU MLP | simpler to express positionally |
| weight tying | untied head | BrittainScript has no attribute assignment (`a.b = c`) |
| `is_causal=True` | explicit bool mask | no `none` literal to pass as `attn_mask` |
| classes | `nn.ModuleList` + integer indexing | BrittainScript has no classes |

Runs at ~19k tok/s on an M3 Max in fp32 (no `autocast` — BrittainScript has no
`with` statement). Interpreter overhead is ~8%: the language re-parses each line on
every loop iteration, but that's negligible against GPU time, since the host
language only *dispatches* work — the FLOPs happen in PyTorch's kernels. Which is
exactly the relationship ordinary PyTorch code has with C++/CUDA.

---

## Measured comparison

`eval_compare.py`, run over the same held-out text (60KB each, 6 samples/prompt):

| model | params | vocab | B/tok | BPB code | BPB prose | syntax valid |
|---|---|---|---|---|---|---|
| BRITTAIN-1 124M (general) | 123.5M | 50257 | 2.05 | 2.060 | **1.506** | 4% |
| brittain2-xs-coder:50m-bs | 51.9M | 32000 | **3.19** | **1.046** | 1.726 | **46%** |

Read this as *specialisation worked*, not "the new one is better":

- **On code the 50M wins decisively** — 1.046 vs 2.060 bits per byte, less than half,
  despite having 2.4x fewer parameters. In compression terms it squeezes Python to
  7.6x versus 3.9x. Syntax validity goes 4% -> 46%.
- **On prose BRITTAIN-1 still wins** (1.506 vs 1.726), exactly as it should — it was
  trained on English and the coder only gets a 15% English mix. That the gap is
  only 0.22 bits/byte is the evidence that the mix did its job: the code model stayed
  literate rather than collapsing into pure syntax.
- **The tokenizer shows up directly**: 3.19 bytes/token vs 2.05, a 56% efficiency gain
  on the same file.

A 52M model beating a 124M model at code, while losing to it at prose, is the
cleanest possible demonstration that the domain choice — not the parameter count —
was what mattered.

(50 generations per model for the syntax figure, identical sampling for both
— temperature 0.4, top_p 0.95, repetition penalty 1.12. Directional, not a
benchmark result.)

**Sampling note.** These small models fall into repetition loops easily. A mild
repetition penalty (~1.12) is essential; without it, low-temperature decoding
degenerates into `x = 0, y = 0, z = 0, ...` forever. The penalty stays mild
because code legitimately repeats — identifiers and indentation recur by design,
so prose-style penalties (1.3+) would fight correct structure.

## Scale, honestly

For calibration on what these can and can't do:

| Model | Params | Tokens |
|---|---|---|
| brittain2-xs-coder | 52M | 1B |
| BRITTAIN-1 | 124M | 2.6B |
| BRITTAIN-2 | 235M | 14.7B |
| GPT-2 XL | 1.5B | ~9B |
| LLaMA-3.2-3B | 3B | ~9T |
| Frontier models | 100B+ | 10T+ |

Models in this repo are **3–5 orders of magnitude** below anything you'd use as an
assistant, on both axes. They produce fluent, plausible text and — for v2 — plausible
code. They do not reason, do arithmetic, or reliably state facts. That's a property
of scale, not of a bug.

These were built to understand the full modern LLM pipeline end to end —
architecture → tokenization → pretraining → instruction tuning — by building every
stage. That goal is met.
