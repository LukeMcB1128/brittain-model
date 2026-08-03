# The BRITTAIN models

Every model here is a **decoder-only transformer** trained from scratch — no
pretrained weights, no fine-tuning of someone else's model. This document explains
the shared architecture, how it differs from GPT-2, and what each checkpoint is.

## Lineage

Two different relations, and conflating them is easy:

- **`└──` solid** — *continued from these weights*. A real descendant: the parent
  checkpoint was loaded and training carried on.
- **`┄┄` dashed** — *a new generation*. Trained from scratch with a different
  architecture, tokenizer, or corpus. Shares ideas, not parameters.

```
BRITTAIN · 604M MoE                    abandoned — char-level, learned positions,
│                                      Python-loop MoE. The context cliff at 32/256
│                                      was structural; nothing carried forward.
│
├┄┄ rewritten: RoPE, SwiGLU, fused attention
│
├── BRITTAIN-1 · 124M            gpt2 BPE 50257 · FineWeb-Edu 2.6B · val 3.247
│   └── 124M Instruct            SFT on Alpaca — the only true fine-tune here
│
├┄┄ retokenized (32k code BPE) + recorpused (The Stack)
│
└── BRITTAIN-2
    │
    ├── XS-Coder · 50M           trained BY BrittainScript (brittain_script/train_50m.bs), on the Mac
    │   └── XS-Specialist        complete · continued for 3 epochs on the mixed
    │                            native + verified-translation BrittainScript corpus
    │
    └── Coder · 235M             1K ctx · 14.7B tokens · val 1.4177   <- the base
        └── + FIM                complete · +2.2B FIM tokens · vocab 32000 -> 32003
            └── + 2K context     planned
                └── + 4K context planned
```

The context steps are a **chain, not siblings** — each is continued pretraining
from the previous checkpoint, so a gain can't be attributed to one change alone.
That's the deliberate trade: the 50M ablations are a far cheaper place to isolate
variables than $14 a time on the 235M.

**Anything downstream of FIM uses a different tokenizer**
(`tokenizers/brittain2-code-32k/tokenizer_fim.json`, vocab 32003). Checkpoints
record their own tokenizer metadata and `src/brittain/tokenizer.py` resolves the
older training-time path, so base and FIM checkpoints can be served side by side.

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
models use), implemented in `src/brittain/model.py`:

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

The SFT stage (`scripts/train/sft.py`) is the same thing that turns a base model into a
chat model: train on instruction→response pairs wrapped in a fixed template, with
the **loss masked over the prompt** so the model learns to *answer* instructions
rather than generate them.

### BRITTAIN-2 — 235M coder *(complete)*

| | |
|---|---|
| Parameters | 235,176,960 |
| Shape | 16 layers, 16 heads, 1024 embd, 1024 context |
| Tokenizer | **custom 32k code BPE** (`tokenizers/brittain2-code-32k/tokenizer.json`) |
| Data | The Stack (Python/JS/TS) + 15% FineWeb-Edu, ~14.7B tokens |
| Hardware | 1× NVIDIA L4, ~7.4 days, ~$135 |
| **Final val** | **1.4177** at iter 23,200 (`brittain2_235m_weights.pt`) |

Landed just outside the projected 1.30–1.40 band.

**The run converged early.** Best val came at iter 23,200 of 28,000; the final
~4,800 iterations (~17% of the run, ~$28, ~1.5 days) never beat it — val bounced
1.42–1.46 with no new low. Train and val stayed within ~0.01 of each other, so
there is no visible train/validation gap. That rules out obvious memorization, but
it does not by itself prove whether the remaining limit is the corpus, model
capacity, or schedule. Future runs should stop after ~2,000 iterations with no new
best.

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
| Trained by | **`brittain_script/train_50m.bs` — BrittainScript** |

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

### brittain2-xs-brittainscript-specialist:50m-bs — 52M *(complete)*

The selected specialist is `brittain2_xs_bs_mixed.pt`, continued for three
epochs from the XS coder. Its corpus mixes native BrittainScript with verified
Python-to-BrittainScript translations at the same token budget used by the
native-only comparison.

| Measurement | Result |
|---|---:|
| Human-written BrittainScript BPB | **0.922** |
| Prompted syntax / runtime | **96.7% / 91.7%** |
| Translation syntax / runtime | **86.7% / 83.3%** |
| HumanEval / HumanEval+ | **0% / 0%** |

The mixed checkpoint was selected over the native-only checkpoints because it
had the best held-out BrittainScript BPB and the strongest translation runtime.
It is a language specialist, not a general Python model.

### brittain2-coder:235m-fim-1k — 235M *(complete)*

Continued from the 235M base checkpoint for approximately 2.2B FIM-formatted
tokens. Three sentinel tokens extend the tokenizer from 32,000 to 32,003. The
model sees the prefix and suffix before generating the missing middle.

| Measurement | Result |
|---|---:|
| Code / prose BPB | **0.737 / 1.254** |
| HumanEval p@1 / p@10 | **2.3% / 6.1%** |
| HumanEval+ p@1 / p@10 | **2.1% / 5.5%** |
| Suffix-required variable named | **71%** |
| Identical output after suffix change | **0%** |
| Hole overrun | **25%** |

The suffix test matters because lower FIM validation loss alone would not prove
that the model reads the code after the cursor. It does. Ending the completion at
the correct boundary is still the obvious weakness.

---

### brittain2-general:254m — 254M *(shelved)*

| | |
|---|---|
| Parameters | ~253,900,000 |
| Shape | 16 layers, 16 heads, 1024 embd, 1024 context |
| Tokenizer | GPT-2 BPE, vocab 50257 |
| Data | FineWeb-Edu, ~12B tokens |
| Iterations | 23,000 x 524,288 tok |
| Hardware | 1x NVIDIA L4, ~6.6 days, ~$113 @ $0.71/hr |
| Projected val | **~2.75-2.90** |

**The transformer is byte-identical to the 235M coder** — 16 layers, 16 heads,
1024 embd, same RoPE/SwiGLU/tying. Only the tokenizer and corpus change. The
parameter difference (254M vs 235M) is entirely the embedding table: 50257 rows
instead of 32000.

This would have served two purposes at once:

**1. An elevation of BRITTAIN-1.** 2x the parameters, 4.6x the tokens, and the
*same tokenizer* — so its val loss is directly comparable to v1's 3.247 with no
BPB conversion. BRITTAIN-1 sat at exactly Chinchilla-optimal (21 tok/param),
which by current practice is undertrained; this run sits at 47.

**2. The controlled experiment.** Same transformer, same order of token budget,
only the corpus differs. That isolates *data* as the variable and lets the
specialisation claim be made properly, rather than inferred from a comparison
that also varies size, tokenizer, and architecture.

It was shelved before training when the remaining budget moved to FIM and the
BrittainScript specialist. The design is kept here as a record, not as part of
the active release plan.

**Why GPT-2's tokenizer and not a custom one:** the 44% win on the code BPE came
from indentation merges, which prose doesn't contain. GPT-2's BPE was itself
trained on English web text, so it's already well matched — a custom English BPE
would buy maybe 5-10%, and dropping to 32k vocab would cost a little packing
efficiency on English. The direct comparability with BRITTAIN-1 is worth more.

## The first BRITTAIN-2 release side by side

`scripts/evaluate/compare.py` produced the BPB rows over identical frozen text.
HumanEval used 164 problems and 10 samples per problem.

| | BRITTAIN-1 | XS coder | XS specialist | 235M base | 235M FIM |
|---|---:|---:|---:|---:|---:|
| parameters | 123,551,232 | 51,917,824 | 51,917,824 | 235,176,960 | 235,180,032 |
| context | 1,024 | 512 | 512 | 1,024 | 1,024 |
| tokenizer vocab | 50,257 | 32,000 | 32,000 | 32,000 | 32,003 |
| training exposure | 2.6B | 1B | 1B + 3 specialist epochs | 14.7B | 14.7B + 2.2B FIM |
| BPB code | 2.031 | 1.080 | 1.121 | 0.751 | **0.737** |
| BPB prose | 1.354 | 1.702 | 1.718 | 1.259 | **1.254** |
| HumanEval p@1 / p@10 | 0 / 0 | 0 / 0 | 0 / 0 | 0.1% / 0.6% | **2.3% / 6.1%** |
| HumanEval+ p@1 / p@10 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **2.1% / 5.5%** |

Raw validation loss is only comparable within a tokenizer. BPB is the cross-tokenizer
comparison used for the release.

### What the numbers say

**The 52M cuts code BPB almost in half from BRITTAIN-1** (1.080 vs 2.031) while
being less than half its size, and loses to it on prose (1.702 vs 1.354). That is
the specialization result in the cleanest form: the domain choice moved the small
model strongly toward code. It does not mean the 52M is broadly better at coding;
both models still score zero on HumanEval.

**The XS specialist makes the intended trade.** Its general code and prose BPB
move slightly backward, while human-written BrittainScript BPB improves from
1.102 to 0.922 and translation runtime reaches 83.3%.

**The 235M coder wins on BOTH axes — including prose.** Code BPB 0.751 against the
52M's 1.080 is a 30% improvement. The surprise is prose:
**1.259 beats BRITTAIN-1's 1.354**, even though the coder saw only 15% English and
BRITTAIN-1 saw nothing else. 2.2B English tokens in a 235M model beat 2.6B English
tokens in a 124M one. The code-heavy corpus did not prevent the larger model from
improving on the prose fixture.

**FIM is the first checkpoint with measurable functional performance.** Through
its intended FIM prompt format it reaches 2.1% HumanEval+ pass@1 and 5.5% pass@10.
It also conditions on the suffix: changing the code after the cursor changed the
completion, and the middle named the suffix-required variable 71% of the time.

**Syntax validity is the honest miss.** 60% against a projected 75-85%. The model
writes code that looks right and often doesn't parse — consistent with what it
does by hand, where it produces plausible-but-wrong statements in the correct local
idiom. This is the clearest single number showing the 235M/14.7B ceiling.

### The honest caveats

- Syntax-validity figures come from **750 generations per model** (150 samples x 5
  prompts), at identical sampling for every model.
- **The code BPB sample is frozen** at `benchmarks/prompts/code.py`, a snapshot of
  `src/brittain/model.py`
  taken 2026-07-31. It used to default to the live model module, so BPB moved
  whenever the architecture was edited — adding the KV cache shifted the 235M from
  0.693 to 0.751 with no change to the model. Figures before that fix are not
  comparable to the release numbers.
- The 235M's val flattened hard after iter ~12,000 (1.532 -> 1.512 across 2.6B
  tokens) and bottomed out at **1.4177 (iter 23,200)**, never improving across the
  final 4,800 iterations.
- HumanEval used 10 stochastic samples per task. The FIM checkpoint was prompted
  through the FIM interface, while the base checkpoints were prompted as normal
  left-to-right models.
- FIM termination is not solved. In the suffix-conditioning test, 25% of samples
  ran past the requested hole into another definition.

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
