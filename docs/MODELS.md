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
│   └── 124M Instruct            SFT on Alpaca — the only true fine-tune of that line
│
├┄┄ retokenized (32k code BPE) + recorpused (The Stack)
│
├── BRITTAIN-2
│   │
│   ├── XS-Coder · 50M           trained BY BrittainScript (brittain_script/train_50m.bs)
│   │   ├── XS-Specialist        continued on native + verified-translation BrittainScript
│   │   └── 50m-bs-4b            4x Chinchilla rerun — see the warning below
│   │
│   └── Coder · 235M             1K ctx · 14.7B tokens · val 1.4177   <- the base
│       └── + FIM                +2.2B FIM tokens · vocab 32000 -> 32003
│           └── + 2K context     +1.42B · SHIPPED as brittain2-coder:235m-fim-2k
│               └── + code SFT   32,849 examples · brittain2-coder:235m-instruct-2k
│
├┄┄ new architecture: GQA, QK-norm, RMSNorm, 24k tokenizer, rope_theta 100000
│
├── BRITTAIN-3
│   ├── 49M pilot                go/no-go · REQUIRES repo/file prompt framing
│   ├── 49M curriculum probe     quality-first corpus probe
│   └── 181M                     PLANNED · gated on corpus + evaluations
│
└┄┄ same BRITTAIN-3 engine, different domain: narrative prose, 8k tokenizer
    │
    └── BRITTAIN-SHAKESPEARE
        ├── 18M pilot            1,500 updates at 1K context
        └── 80M                  TRAINING · tag-conditioned story model
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

## What BRITTAIN-3 changes again

The 49M pilot, the curriculum probe, the planned 181M, and both Shakespeare
models share a second-generation engine (`src/brittain/model_v3.py`). Four
changes from the BRITTAIN-1/2 block:

| | BRITTAIN-1/2 | BRITTAIN-3 |
|---|---|---|
| attention | multi-head | **grouped-query (GQA)** |
| Q/K scaling | none | **QK-norm** |
| normalisation | LayerNorm | **RMSNorm** |
| RoPE base | 10,000 | **100,000** |

**GQA** shrinks the KV cache by the query-to-KV head ratio — 3x on the 49M
(9 query heads, 3 KV), 2x on the Shakespeare 80M (10 and 5). That is what makes a
16K maximum context affordable to serve at all.

**QK-norm** normalises queries and keys before the attention product. It
stabilises training at higher learning rates, which shortens schedules.

**RMSNorm** drops the mean-subtraction and the bias of LayerNorm for equivalent
quality at lower cost.

**rope_theta 100000** stretches the rotary period so positions stay
distinguishable at long context, rather than aliasing the way a 10,000 base does
past a few thousand tokens.

`intermediate_size` is also stated explicitly per config rather than derived from
`8/3 × d`, so width and depth can be traded without the MLP silently resizing.

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


### 4K context — dropped

4K belongs in a model designed for it, not bolted onto a 1K model. It also has a
real downside here: roughly 45% of 4K windows would span unrelated files, and
there is no document-boundary attention masking. Extending context without that
masking partly trains the model to ignore its own context.

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

### brittain2-coder:235m-fim-2k — 235M *(complete, shipped)*

| | |
|---|---|
| Parameters | 235,180,032 |
| Shape | 16 layers, 16 heads, 1024 embd, **2048 context** |
| Tokenizer | code BPE 32003 (`tokenizer_fim.json`) |
| Data | +1.42B tokens on top of the FIM checkpoint |
| Cost | ~$12.6 on an L4 |
| Checkpoint | `brittain2_235m_fim_2k.pt` |

The autocomplete model. 2048 tokens is roughly 195 lines of code against ~100
before, so it can hold a whole small module rather than the neighbourhood of the
cursor.

**Ship the ANNEALED final, not `_best`.** Validation said the opposite — 1.3045 at
iter 1000 against ~1.36 at the end — and validation was wrong. The annealed
checkpoint won every BPB metric and +6 points of syntax validity. A mid-cosine
"best val" checkpoint has hot weights; finish the anneal and decide on BPB.

HumanEval was **flat** against the 1K FIM model (2.44% vs 2.26% pass@1), exactly
as predicted: those prompts are a few hundred tokens and cannot use a 2048 window.
Longer context is for real files, not for benchmarks made of short ones.

### brittain2-coder:235m-instruct-2k — 235M *(complete, shipped)*

| | |
|---|---|
| Parameters | 235,180,032 |
| Shape | 16 layers, 16 heads, 1024 embd, 2048 context |
| Tokenizer | code BPE 32003 — inherited from its FIM base |
| Data | 32,849 SFT examples (19,849 CodeAlpaca, 10,000 Magicoder, 3,000 alpaca-cleaned) |
| Cost | ~$1 on an L4 |
| Checkpoint | `brittain2_235m_instruct_2k.pt` |

The first BRITTAIN model that answers a question about code rather than
continuing it. The SFT was never about raw capability — it was about **knowing
when to stop**. Runaway generation fell 5.4x, 97% to 18% of completions hitting
the token cap under greedy decoding.

**It needs the Alpaca template.** Handed a bare instruction it continues the
sentence instead of answering. `scripts/inference/chat.py` and the server's
instruct mode apply it; `sample.py` deliberately warns rather than templating.

**BPB cannot judge this model.** It scores *worse* on code BPB than its base
(0.851 vs 0.687) because it now models Alpaca-formatted instructions rather than
raw source. Selecting on BPB would systematically prefer whichever checkpoint
fine-tuned least — the one that did the least work. It also keeps the FIM
sentinels from its base, so `supports_fim` stays true while `mode` is `instruct`;
the server treats those as independent for that reason.

### BRITTAIN-3 — 49M pilot and curriculum probe *(complete)*

| | |
|---|---|
| Parameters | 49,558,592 |
| Shape | 10 layers, 9 heads, **3 KV heads**, 576 embd, intermediate 1536 |
| Tokenizer | `tokenizers/brittain3-code-24k`, vocab 24,576 |
| Context | 2048 trained, **16384 maximum** |
| Checkpoints | `checkpoints/brittain3_49m_pilot/`, `checkpoints/brittain3_49m_curriculum_probe/` |

A go/no-go pilot for the planned 181M, not a release model.

**Prompt framing is not optional.** Every pretraining document was wrapped as
`<|repo_start|>{repository}<|file_start|>{path}\n`. Measured on this checkpoint,
an unframed prompt returns `<|file_end|><|repo_end|>` and stops after two tokens,
every sample. `serve.py` applies the framing server-side.

The pilot carries a `card.json` beside the checkpoint because it predates the
checkpoint payload carrying a corpus block. Anything trained after it should embed
that metadata in the checkpoint, where it cannot drift from the weights.

### brittain3-coder:181m — 181M *(planned)*

The real capability target: a model that writes working simple code at a
novice-developer level and operates the Brittain app's agent tools.

**Gated.** Do not start the paid L4 run until the focused corpus and the
novice-code evaluation suite exist. The reasoning is in this document's findings —
Brittain2 saw 14.7B code-heavy tokens and still fails basic instructions, so
scaling alone will not fix it. Quality-first corpus, verified by execution, or the
run repeats the same ceiling more expensively.

### brittain-shakespeare — 18M pilot and 80M *(80M training)*

A narrative-prose model on a separate branch, sharing the BRITTAIN-3 engine
unmodified. Trains on a single RTX 3060 12GB.

| | 18M pilot | **80M** |
|---|---|---|
| Parameters | 18.1M | 80M |
| Layers / heads / KV | 8 / 6 / 2 | 16 / 10 / **5** |
| Embedding / intermediate | 384 / 1280 | 640 / 1792 |
| Context | 1024 | **4096** |
| Tokenizer | `brittain-shakespeare-prose-8k`, vocab 8192 | same |

**Tokenizer.** 8192 vocab trained on 900MB with `archaic_boost: 3`. Efficiency by
register: modern prose 3.67 bytes/token, early modern 3.52, dialogue 3.13, tag
blocks 2.50.

**Corpus.** ~1.5B target tokens — 1.2B Gutenberg fiction, 120M Standard Ebooks,
48M early modern (**upsampled 4x**), 80M world texture, 60M synthetic, with 40M
reserved for a later SFT pass. Story prose only: essays, treatises, reference
works and biography are excluded, because at this parameter count there is no
capacity to spend on knowledge that never appears inside a story.

**Schedule.** Four stages, 13,734 updates x 131,072 tokens = **1.80B tokens**,
~22 tokens/parameter. Context grows 1K to 2K to 4K, then a 4K anneal. Measured on
the 3060: 17,137 tok/s at 1K, 15,630 at 2K, 9,887 at 4K, peak 7.0GB — about
33 hours end to end.

**Nine conditioning tags** with closed value vocabularies, prepended to every
pretraining document:

```
<|story_start|><|tags|>[Voice: Modern] [Genre: Tragedy] [Setting: Tavern]<|end_tags|>
```

`Voice`, `Genre`, `POV`, `Tense`, `Setting`, `Tone`, `Cast`, `Length`, `Twist`.
Because the tags are present from the first pretraining token they are native
structure, not a post-hoc instruction layer, and by the end of training they act
as control levers.

Two design decisions carry the project. **Labels are correct by construction** —
eight of nine tags are computed from the text or its bibliographic metadata by
regex and lexicon, no model and no API. A tag the model cannot verify against the
text teaches it that tags are noise, and the lever goes dead. And **the same
tagger scores generated samples**, so tag adherence is a measured number rather
than an opinion — the oracle problem that prose usually lacks.

One trap worth recording: **`Voice` cannot come from publication year.**
Gutenberg's `dcterms:issued` is its own release date — Dracula is stamped 1995 —
so deriving register from it would have labelled nearly the whole corpus `Modern`
and trained the lever on noise, while every test still passed. `Voice` comes from
archaic morphology in the text, falling back to the author's death year.

## Every model, measured

BPB is over identical frozen held-out text (`data/eval_code.py`, `english.txt`).
Lower is better, and it IS comparable across tokenizers — raw validation loss is
not. HumanEval is 10 samples/task at temperature 0.4 / top_p 0.95 / rep 1.12.

| model | params | shape | ctx | tokens | BPB code | BPB prose | syntax | p@1 | p@10 |
|---|---|---|---|---|---|---|---|---|---|
| 604M MoE | 604M | char-level MoE | 32 | ~3M | — | — | — | — | — |
| `124m_best` | 124M | 12L/12H/768 | 1024 | 2.6B | 2.031 | 1.354 | 0% | 0.00% | 0.00% |
| `124m_sft` | 124M | 12L/12H/768 | 1024 | +Alpaca | — | — | — | N/A | N/A |
| `235m_weights` | 235M | 16L/16H/1024 | 1024 | 14.7B | 0.751 | 1.259 | 67% | 0.06% | 0.61% |
| `235m_fim` | 235M | 16L/16H/1024 | 1024 | +2.2B | 0.737 | 1.254 | 56% | 2.26% | 6.10% |
| **`235m_fim_2k`** | 235M | 16L/16H/1024 | **2048** | +1.42B | **0.687** | **1.233** | 62% | **2.44%** | 5.49% |
| **`235m_instruct_2k`** | 235M | 16L/16H/1024 | 2048 | +32.8k ex | 0.851 | 1.554 | — | N/A | N/A |
| `50m_bs` | 52M | 6L/8H/512 | 512 | ~1B | 1.080 | 1.702 | 53% | 0.00% | 0.00% |
| `50m_bs_4b` | 52M | 6L/8H/512 | 512 | ~4B | **1.002** | **1.603** | **38%** | 0.00% | 0.00% |
| `xs_bs_mixed` | 52M | 6L/8H/512 | 512 | +BS | — | — | — | 0.00% | 0.00% |
| `49m_pilot` | 49.6M | 10L/9H/3KV/576 | 2048 | — | — | — | — | — | — |
| `181m` | 181M | planned | — | ~5.2B | — | — | — | — | — |
| `shakespeare_18m` | 18.1M | 8L/6H/2KV/384 | 1024 | pilot | — | — | — | — | — |
| `shakespeare_80m` | 80M | 16L/10H/5KV/640 | 4096 | 1.80B | — | — | — | — | — |

Runaway generation — completions hitting the token cap instead of stopping,
greedy over 60 HumanEval tasks:

| model | ran to the cap |
|---|---|
| `235m_fim_2k` | 58/60 — **97%** |
| `235m_instruct_2k` | 11/60 — **18%** |

Do not compare syntax percentages taken at different sample counts. It is a
binomial proportion: at ~100 samples sigma is about 5 points, which is why the
1B BrittainScript checkpoint reads 53% here and 46% in older notes.

## More data compressed better and generated worse

The sharpest single result in the project. Same 52M model, same corpus, 4x the
tokens:

| checkpoint | tokens | BPB code | syntax |
|---|---|---|---|
| `brittain2_50m_bs` | ~1B | 1.080 | 53% |
| `brittain2_50m_bs_4b` | ~4B | **1.002** | **38%** |

BPB improved 0.078 and syntax validity fell 15 points. **The two moved in
opposite directions.** HumanEval stayed at 0.00% for both, so the benchmark could
not see a difference that BPB says is real — which is its own finding: HumanEval
has no resolution below roughly 200M parameters.

This is the core argument for a quality-first corpus in BRITTAIN-3. More tokens of
mixed-quality data buys compression while degrading usable output.

## `_best` picked the wrong checkpoint three times out of three

Every training script writes a `_best.pt` on the lowest validation loss. On three
consecutive runs, measured on the capability the run actually existed to buy, it
was the worse checkpoint.

| run | `_best` said | what the capability test said |
|---|---|---|
| FIM | val 1.3604 at iter 1800 | ran past the hole **44%** of the time vs **17%** for the annealed final |
| 2K context | val 1.3045 at iter 1000 | lost **every** BPB metric and 6 points of syntax to the final |
| code SFT | BPB 0.767 | truncated **32%** of the time vs **18%** for the final |

Three different runs, three different metrics, one conclusion: **finish the
anneal, then decide on the capability you want, not on validation loss.** A
mid-cosine checkpoint has hot weights; low validation loss at that point measures
a model still in motion.

The SFT case is the sharpest, because BPB there is actively misleading. An
instruction tune *should* score worse on raw-source BPB — it now models
Alpaca-formatted text. Selecting on BPB systematically prefers whichever
checkpoint fine-tuned least.

Every shipped checkpoint in this document is a final. `_best.pt` files are kept as
crash-recovery artifacts, not as release candidates. Several also remain on disk
purely as dev history — `brittain2_235m_2k_best_weights.pt`,
`brittain2_235m_instruct_best.pt`, `brittain_50m_bs_expanded_best.pt`,
`xs_bs_native.pt`, `xs_bs_native6.pt` — none are released, and
`brittain_model_backup.pt` is the 604M prehistory model, which current code
cannot load.

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
