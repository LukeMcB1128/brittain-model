# Brittain Live — a coding model that learns while it works

Working name. Status: **plan, nothing built yet** (written 2026-10-05).

Every coding model today is frozen: what it knows about *your* project has to
fit in the prompt, and it forgets when the session ends. Brittain Live keeps
learning at inference time. As it reads a repo, runs tests and sees errors, a
neural memory inside the model updates its own weights. The memory state can
be saved to a file (~35 MB at pilot scale) and reloaded the next day.

The claim to prove, in one chart: **on a library it has never seen, larger than
its attention window, Brittain Live's pass rate rises over a session; a frozen
model of the same size stays flat; and the gain survives clearing the context
and reloading the saved memory.**

This document covers the 3060 pilot only. It decides whether the 1B
(`brittain-1b` credits plan) gets memory layers. That decision has to be made
before the 1B starts, because a resume cannot change the architecture.

## Prior art this builds on

- **Titans** (Google, 2025): a neural long-term memory, an MLP whose weights are
  updated during the forward pass by a gradient step on an associative loss,
  with momentum ("surprise") and weight decay ("forgetting"). Memory runs
  alongside sliding-window attention.
- **HOPE / Nested Learning** (Google, NeurIPS 2025): a self-modifying Titans
  variant with memory levels that update at different rates.
- **Code test-time training**: PonderTTT (GPT-2-scale TTT on code), Code2LoRA
  (per-repo adapters from a hypernetwork). Neither is a from-scratch coder whose
  core design is learning from execution feedback.

What is new here: execution feedback (test results, tracebacks) as the signal
that drives memory writes, trained into a coding model from scratch, and a
benchmark that isolates the memory's contribution.

## The lesson from the 49M pilot applies

The Brittain3 49M pilot reached 0.8% novice pass@1. A ~60M model will not
write real programs. So the pilot **does not test program synthesis**. It tests
whether the model can *learn an API it has never seen* and use it correctly in
short completions, where a 60M model can score well above zero. Program
synthesis is the 1B's job.

## Architecture

Start from `src/brittain/model_v3.py` (RoPE, SwiGLU, GQA, QK-norm, RMSNorm) and
add a new architecture value, `brittain_live`, rather than a flag on
`brittain3`, so old checkpoints and configs stay valid.

Pilot shape (both arms share it):

| | value |
|---|---|
| layers | 12 |
| width | 512 |
| heads | 8 query / 2 KV |
| SwiGLU hidden | 1376 |
| attention | **sliding window, 1,024 tokens** |
| train sequence length | 8,192 |
| embeddings | tied |
| total params | ~58M at a 49,152 vocab (~33M non-embedding) |

The sliding window is the important part. With full attention over 8K, the
model can ignore the memory, and the experiment proves nothing. With a 1K
window, 7/8 of every training sequence is visible *only* through the memory.

**Memory module** (Titans "memory as gate", MAG, in layers 3, 6, 9, 12):

- Memory `M` is a 2-layer MLP, 512 → 1024 → 512. Its weights are per-sequence
  state, initialised from learned `M_0`.
- Per chunk of 64 tokens: compute keys `k` and values `v` by projection. The
  surprise is the gradient of `||M(k) − v||²` with respect to `M`. Update with
  data-dependent learning rate `η_t`, momentum `θ_t` and decay `α_t`, each a
  small linear map of the input, followed by a sigmoid.
- Read: `y = M(q)`, mixed into the attention output by a learned gate.
- Chunk-parallel update within a chunk, sequential across chunks, as in the
  Titans paper. Write a slow sequential reference implementation first, and
  unit-test that the parallel version matches it.
- Saved state per memory layer is the MLP weights plus momentum: 4 layers ×
  ~1.05M × 2 × fp32 ≈ 34 MB.
- Gradients through 128 chunks of memory updates are memory-hungry. Use
  truncated BPTT (detach the memory state every N chunks) if 12 GB isn't
  enough. Measure first.

**Feedback gate (arm B2, only after B1 works):** add a segment-type embedding
(code / docs / test output / traceback) and let `η_t` depend on it, so the
model can learn to write harder on errors. v1 just puts feedback in the token
stream and lets the ordinary surprise signal handle it.

Cross-check the memory implementation against an open Titans
reimplementation before trusting any numbers.

## Arms

| arm | what | why |
|---|---|---|
| **A** | SWA-1K transformer, param-matched (add a layer) | the frozen baseline |
| **B1** | SWA-1K + neural memory | the claim |
| B2 | B1 + feedback-gated writes | the twist, only if B1 passes |
| C (optional) | full-attention 8K transformer | reference ceiling within context |

Same data, tokens, seed and schedule for A and B1.

## Data

The memory only learns to matter if training sequences contain long-range
dependencies. Short files packed together teach it nothing.

1. **Repo-level packing (~70%).** Concatenate the Python files of one repo in
   import/dependency order into 8K sequences, so later files call functions
   defined 5K tokens earlier. Reuse the existing code pipeline
   (`scripts/prepare/prepare_code.py`, the decontamination step), but pack by
   repo rather than by file.
2. **Execution-feedback documents (~20%).** Synthetic: code → run → test
   output or traceback → fix → run again. Generated locally by executing
   snippets in a sandbox. The execution-verified exercise tooling
   (`build_exercises_v3.py`, `verification_v3.py`) is the starting point.
3. **Generic Python/English (~10%)** so the model stays a language model.

Keep a manifest of exactly what went in. Every "it learned this" claim depends
on proving the library wasn't in pretraining.

**Budget:** ~1.5B tokens per arm. Measure tokens/sec on the 3060 before
committing. Estimate from the Mac's measured 49M rate × 3: ~18–25K tok/s for
arm A, so ~20 h; arm B1 ~1.5–2× slower, so ~30–40 h. Roughly **3 days of 3060
time for the gate.**

## The benchmark: UnseenLib

A procedural generator for fake Python libraries that cannot exist in any
pretraining data.

- Random module, class and function names. Signatures with non-obvious argument
  order, keyword-only parameters and custom exception types. Semantics composed
  from templates (e.g. `scale_ring(values, *, pivot)` rotates then scales), so
  the behaviour can't be guessed from the name.
- Real source and docstrings, **30K+ tokens per library**, far beyond the 1K
  window.
- Hidden tests for every task, run in a sandbox.

Task tiers, sized for a 60M model:

- **Tier 0, API recall:** complete `result = lib.` with the right function,
  argument order and keywords. Scored by execution.
- **Tier 1, one-liners:** one or two lines that need the function's
  *semantics*, not just its name.
- **Tier 2, small functions:** reported, not gated on, at this scale.

Session protocol: stream the library source → tasks in order → for each task,
generate, run the tests, append the result or traceback, then move on. The
memory updates throughout. Held-out libraries only, never seen in training.

Conditions per library (all scored on the same tasks):

1. **A, frozen:** only the last 1K tokens are visible.
2. **A + BM25 retrieval:** the top library chunks are put into the window.
   **This is the real competitor.** RAG is what people use today.
3. **B1, live:** memory updates on.
4. **B1, memory reset before tasks:** proves the memory does the work.
5. **B1, memory frozen after reading:** separates "learned from reading" from
   "learned from feedback".
6. **B1, save → fresh process → reload:** the persistence demo.

Also a diagnostic run early and often: **S-NIAH-style key-value recall** at
distances of 1K–32K tokens. It's cheap, and it's the first thing to break.

**Transfer metric.** After the model fails task *i* using API *f* and sees the
traceback, what's its pass rate on *later* tasks that use *f*? A retry
succeeding is in-context learning. A later, different task succeeding is the
weights learning.

## Gate (go/no-go for putting memory into the 1B)

On held-out libraries, Tier 0 + Tier 1:

1. B1 beats A-frozen by **≥15 points absolute** by the end of the session.
2. B1 with memory reset falls to within 3 points of A. Otherwise the gain
   isn't coming from memory.
3. The saved-and-reloaded memory keeps **≥80%** of the gain.
4. B1 **matches A + BM25**. Beating it is what makes the project worth going
   public with. Matching it with no retrieval system and a portable memory
   file is still a result.
5. On repo-level validation data, B1's loss at positions beyond the window is
   clearly below A's. This is the diagnostic that should move first.

If 1–3 pass but 4 fails: the architecture works, but the pitch needs more
scale. Repeat at ~150M before deciding.
If 1 or 2 fail: write it up, keep the code, and go ahead with the 1B as a plain
transformer.

## Phases

| # | phase | output | rough time |
|---|---|---|---|
| 0 | Memory module + sequential reference + parity tests; SWA via flex attention; memory save/load | `model_live.py`, tests | 1–2 weeks |
| 1 | Repo-level packer, execution-feedback generator, UnseenLib generator + sandboxed harness, BM25 baseline | data + `benchmarks/unseenlib/` | 1–2 weeks |
| 2 | Smoke runs: ~15M, 200M tokens, A vs B1, S-NIAH probe. **Kill switch:** if memory can't beat SWA on recall beyond the window, fix it before spending 3 days | probe curves | 2–3 days |
| 3 | Gate runs: A and B1 at ~58M, 1.5B tokens | checkpoints | ~3 days of 3060 time |
| 4 | UnseenLib eval, all six conditions, the chart | results + write-up | ~1 week |
| 5 | Decision on the 1B; if go, B2 feedback-gating ablation and the ~150M confirmation run | decision note | — |

About 5–7 weeks of calendar time, $0 of cloud spend.

## Risks

- **The model ignores the memory.** Mitigated by the 1K window and repo-level
  packing. The S-NIAH probe catches it in phase 2.
- **Training instability.** Learned inner learning rates can explode. Clamp
  `η_t`, use warmup, log memory-weight norms.
- **Throughput.** Sequential chunk updates are slow in pure PyTorch. Use
  `torch.compile` first; custom kernels only if the gate passes.
- **RAG wins.** Then the honest result is "no better than retrieval at 60M",
  and the persistence + no-infrastructure angle has to carry it, or the scale-up
  has to beat it.
- **Overwriting.** A long session may erase early knowledge. Track Tier 0
  accuracy on the session's *first* APIs at the end of the session.
- **Someone ships it first.** This area is moving. Phases 0–2 should be quick.

## Open decisions

- **Tokenizer.** The 1B plan uses a 49,152 vocab. At 58M, that embedding is
  ~43% of the params. Use the 1B tokenizer anyway if it exists, so results
  carry over; otherwise train it first.
- **Name.** "Brittain Live" is a placeholder.
