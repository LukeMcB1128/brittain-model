# brittain3-coder:181m — the capability plan

Goal: the most capable model this project can train on owned hardware. Not a
frontier assistant — a small model that, given a plain instruction, writes
working novice-level code far more often than anything in `docs/MODELS.md`.
The product is an **SFT'd, promptable instruct model**, not a FIM autocomplete
model. FIM is dropped from every stage below; the only FIM artefact that
survives is that the strongest existing base happens to carry FIM sentinels.

## Where this starts: the 49M pilot failed its gate

The pilot (`docs/BRITTAIN3_PILOT.md`) cleared both BPB bars (1.020 code, 1.622
prose at a matched 512 window) and lost the generation half: 3 correct novice
generations out of 360 (0.8% pass@1) against 8/360 (2.2%) for Brittain2 XS 4B.
The 181M run was therefore not started. This plan does not quietly restart it.

What the pilot did and did not show:

- It tested a **filtered real-code** corpus. Verified synthetic was ~1% of it.
  So it is evidence that curated GitHub alone does not buy generation quality
  at this scale — the same lesson as the 235M, which had 14.7B tokens of it.
- It was a 708M-token run scored against a 4B-token baseline. The XS line's own
  trajectory was 0.8% at 1B and 2.2% at 4B; the pilot at 708M landed at 0.8%.
  That is a tie at matched tokens, not a loss, but it is also not the win the
  quality thesis needed. Filtered real code is not the lever.
- The one source the pilot barely used — execution-verified teacher data — is
  the one with outside evidence behind it: phi-1's filtered-web-only base got
  ~29% HumanEval at 1.3B, and the verified synthetic exercise set took it to
  ~50%; phi-1-small (350M) reached ~45% on the same data.

So the bet changes from "quality-first real code" to "verified synthetic at
scale", and it gets tested for $0 before any from-scratch run:

**Step 1 — the lever test (free, ~1 week).** Continue-pretrain the strongest
existing base weights, `brittain2_235m_fim_2k` (the only model with measurable
function: 2.4% HumanEval), on verified synthetic data on the 3060 with a plain
next-token objective — **no FIM data, no FIM loss** — then SFT into an instruct
model and run rejection sampling. Start from the FIM base rather than
`instruct_2k` because continued pretraining on top of an SFT'd checkpoint
partly undoes the SFT; the sentinels in the vocab are inert if never trained
on. It fits at `batch_size=8, grad_accum=64`. If verified synthetic cannot move
a model that already has a foothold, nothing from scratch would have either,
and the project has spent nothing finding out. If it does move it, the result
is a shippable instruct model *and* the justification for step 2.

**Step 2 — the from-scratch 181M**, only if step 1 clears its gate. Keep the
BRITTAIN-3 engine and shape (architecture is second-order here; a new engine
is risk without return). The corpus is synthetic-dominant, real code is
classifier-filtered, the 3060 does the pretraining for free, the anneal runs
on the highest-quality mix, and the $50 of L4 goes only to what the 3060 cannot
do quickly: long-context extension and the SFT/rejection-sampling loop.

The rest of this document specifies both steps.

Targets. The shipped model is prompted through the Alpaca template
(`chat.py` / the server's instruct mode), so the headline numbers are measured
**through the template** with the task stated as an instruction. The base
suites are still run on the pre-SFT checkpoint as the diagnostic for whether
pretraining moved. BPB is a regression alarm only: an instruct model *should*
score worse on raw-source BPB, as `instruct_2k` already showed.

| Metric | Best today | Target |
|---|---:|---:|
| novice_v2 pass@1, base checkpoint (completion prompt) | ~1% | ≥ 25% |
| novice_v2 pass@1, instruct checkpoint (instruction prompt) | — | ≥ 40% |
| HumanEval pass@1, instruct (templated) | 2.4% (FIM-prompted) | 10–20% |
| Syntax validity | 67% | ≥ 95% |
| Runaway to cap (greedy) | 18% | ≤ 5% |
| Answers a bare instruction without continuing it | no | yes, on 95%+ of a held-out prompt set |

## Compute budget, honestly

| Machine | Realized | Role |
|---|---|---|
| RTX 3060 12GB | ~18 TFLOPS bf16 | **Pretraining.** Free, unlimited wall-clock. The workhorse. |
| M3 Max 36GB | 5.9 TFLOPS | **Teacher generation** (Ollama, 30B-A3B MoE fits in unified memory). Also 49M ablations if the 3060 is busy. |
| 3050 laptop 6GB, i9 | small | **CPU jobs**: tokenisation, dedup, classifier training, test execution. Optionally a second small teacher (a 4B dense model at Q4 fits in 6GB). |
| L4, $50 ≈ 70 h | 32.4 TFLOPS | **Context extension + SFT/RFT** only. ~$15 contingency. |

Throughput assumptions (measure before committing — the Shakespeare 80M gave
17.1k tok/s at 1K and 15.6k at 2K on the 3060, so scale by 181/80):

| 181M on the 3060 | tok/s | tokens/day | 8B tokens |
|---|---:|---:|---:|
| act. checkpointing ON (config default) | ~7k | ~600M | ~13 days |
| act. checkpointing OFF (should fit at mb 4–8, 2K) | ~9–10k | ~800M | ~10 days |

`activation_checkpointing: true` is a 30–40% tax; the 49M pilot peaked at 1.6GB
with it on. Turn it off for the 1K/2K stages and only enable it at 8K. Remember
a resume cannot change the model config — decide before starting.

L4 math: 70h × 32.4 TFLOPS ≈ 8e18 FLOPs ≈ 7B tokens at 181M ideal, ~4–5B real.
That is not a lot more than two weeks of 3060 time, which is why pretraining
does not go on the L4.

## Why not a bigger or different model

- **Bigger (≥300M):** trains at half the speed on the 3060 and the gain is
  small compared to the data change. 235M with bad data lost to what 350M with
  good data does. Spend the compute on tokens and quality, not parameters.
- **New architecture:** the v3 engine already has GQA, QK-norm, RMSNorm, RoPE
  θ=100k, SwiGLU, tied embeddings. Remaining architecture wins at this scale are
  small (see ablations). Not worth the risk of a fresh engine.
- **MoE:** was abandoned once for good reasons; on 12GB it buys nothing.

## The corpus — where the capability comes from

Budget: ~9B token-positions seen, over a corpus of ~5B unique tokens.
Three sources, in order of importance:

### 1. Teacher-generated, execution-verified synthetic (the lever)

The pipeline exists (`scripts/prepare/build_teacher_seeds_v3.py`: Qwen3.6 plans,
Qwen3-Coder writes, GLM reviews with independent tests, accept only if both test
sets pass). Today it produces 1% of the curriculum corpus. It needs to become
~15–25% of pretraining tokens and ~100% of the anneal and SFT data.

**Throughput is the real bottleneck, not compute.** Qwen3-Coder measured 71
tok/s single-stream. Run the Mac 24/7 from day one with `OLLAMA_NUM_PARALLEL`
4–8; the MoE should batch to roughly 150–250 tok/s aggregate (measure). At
200 tok/s → ~17M tokens/day → ~500M raw / ~300M accepted per month. Add the
laptop with a 4B dense coder for another ~100 tok/s of *easy* items.

Four synthetic families, each a separate generator on the same accept-if-tests-pass rule:

| Family | What it is | Share of synthetic |
|---|---|---:|
| Exercises | task brief → solution → tests (current pipeline) | 40% |
| Textbook lessons | short concept explanation + worked example + exercise, per topic in the curriculum categories | 25% |
| Repairs | mutate a verified solution (off-by-one, wrong var, missing return), capture the traceback / failing test, teacher writes the fix. Trains "fix this error" | 20% |
| Repo-scale | 2–5 file mini-projects with tests; also provides real multi-file context for the 4K/8K stages | 15% |

Languages: Python 45 / TypeScript 25 / JavaScript 20 / shell+JSON+YAML 10.
Drop Rust/C/C++/Go from synthetic — 181M has no capacity for seven languages,
and the app's agent tools are Python/TS/JS. Real-code slices may keep a little.

Quality rules that keep the lever alive:
- Reject on any test failure; one repair round max (as now).
- Structural-fingerprint dedup so 42 templates cannot dominate (as now).
- Decontaminate against `benchmarks/novice`, `novice_v2`, HumanEval, MBPP
  before packing (`decontaminate_v3.py`).
- Synthetic tolerates repetition better than web data: plan **3 epochs** of
  synthetic within the stable phase, then it dominates the anneal.

### 2. Classifier-filtered real code (the bulk)

~3.5–4B tokens of Python/TS/JS from The Stack, but filtered:

1. Sample 30–50k files. Have the teacher score each 0–3 for "educational value
   for a novice: clear, self-contained, well-named, tested". This is ~1 day of
   teacher time.
2. Train a fastText or small-embedding classifier on those labels (CPU, minutes).
3. Keep the top ~30% by score, plus any file with a companion test file.
4. Strong near-dedup, repo caps, secret scrubbing — all existing.
5. Keep the `<|repo_start|>{repo}<|file_start|>{path}` framing; it works.

Test code + companion implementation stays over-represented (the curriculum
corpus already does this at 40/30) because it is the one real-data source with
an execution signal built in.

### 3. Prose and structured (small, unchanged)

~10% clear English (FineWeb-Edu, high-score slice only), ~5% docs/README,
~5% JSON/YAML/diffs/tool protocol. The tool-protocol tokens are already in the
tokenizer; keep ~0.5% so the SFT stage does not meet them cold.

### No FIM

FIM is out. The product is prompted, not cursor-driven, and the two FIM chains
in `MODELS.md` bought autocomplete behaviour at the cost of 25% hole overrun and
97% runaway — the exact stopping problem SFT has to fix afterwards. Every code
document is trained left-to-right with `<|file_end|>` as the boundary the model
learns to stop at. The atomic FIM tokens in the 24k tokenizer stay unused.

### Instruction data (the SFT set)

Generated by the same teacher pipeline and held to the same accept-if-tests-pass
rule, but shaped as instruction → response in the Alpaca template:

| Type | Share |
|---|---:|
| "Write a function that…" → code (+ the tests it passed, hidden from the model) | 45% |
| "Fix this error" → traceback + broken code → fixed code | 20% |
| "Explain / what does this do" → short prose answer | 15% |
| Multi-step edits to a given file ("add validation, then…") | 15% |
| Plain-English chat that must *not* produce code (so it learns when not to) | 5% |

Target ~100k examples; response-only loss; 20% base-corpus replay to stop the
BPB regression from becoming a capability regression.

## Recipe changes to the trainer (do these before the big run)

1. **Document-boundary attention masking.** Currently absent; ~45% of 4K
   windows span unrelated files, which trains the model to ignore context.
   Implement with FlexAttention block masks (torch ≥ 2.5, works on Ampere) or
   varlen SDPA. This is the one *required* engineering item, and it is what
   makes the long-context stages honest instead of decorative.
2. **Per-stage best checkpoint.** The trainer keeps one `best_validation`
   across stages and clobbered stage 1 in the pilot. Namespace by stage. Finals
   are what ship anyway.
3. **Anneal on the quality mix.** The schedule is already warmup-stable-decay.
   Make the decay stage read from a *different* pack: 60% synthetic / 25%
   top-decile real code / 15% rest. This is the single cheapest quality trick
   known for small models, and it is consistent with every "annealed final beat
   `_best`" result in `MODELS.md`.
4. **Checkpoint metadata in the payload** (corpus block, tokenizer hash,
   relative tokenizer path) — the `card.json` sidecar was a stopgap.
5. Optional, only if the 49M ablation says yes: **Muon** for the 2-D weights
   (AdamW for embeddings/norms). Consistently 1.3–2x sample-efficient in small
   GPT speedruns; risk is implementation, which the 49M run retires for free.

## Schedule

Tokens per update: 524,288 (mb × accum × ctx). Peak LR 6e-4 (QK-norm allows it;
the 5e-4 in the config is fine too), WD 0.1, β=(0.9, 0.95), clip 1.0.

| Stage | Where | Ctx | Tokens | Data | Time |
|---|---|---:|---:|---|---|
| A. stable | 3060 | 2048 | 6.5B | full mix, synthetic ×3 epochs | ~9–11 d |
| B. anneal | 3060 | 2048 | 1.5B | quality mix (above), LR → 0 | ~2–3 d |
| C. extend | L4 | 4096 → 8192 | 0.8B | repo-scale synthetic + multi-file real, doc masking | ~7 h, ~$5 |
| D. instruct SFT | L4 | 4096 | ~60M (×2 ep) | the instruction set, Alpaca template, response-only loss, base replay 20% | ~2 h, ~$2 |
| E. RFT ×2–3 | L4 + Mac | 4096 | ~30M/round | sample k=8 on held-in tasks, keep test-passing, retrain; then DPO on pass/fail pairs | ~4 h/round, ~$3 |
| Contingency | L4 | | | | ~$15 |

Skip the 16K stage. It was 305 updates — a costume, not a capability — and
nothing in the evals can use it. `max_seq_len` stays 16384 so it can be revisited.

Start at 2K rather than 1K: GQA + doc masking make 2K cheap enough, and it
avoids a stage boundary that showed up as a val discontinuity in the pilot.

Stopping rule for stage A: no new best over 2,000 updates → start the anneal
early. Do not let it run to the update count out of stubbornness (the 235M
burned $28 that way).

## Phase plan with gates

**Phase 0 — build (weeks 1–2, Mac + laptop CPU; 3060 finishes Shakespeare).**
Teacher generation starts on day 1 and never stops. It generates in *both*
tokenizers' favour — the same text is packed for Brittain2's 32k FIM tokenizer
(step 1) and Brittain3's 24k (step 2). Build the classifier, the filtered
real-code pack, the doc-masking + per-stage-best trainer changes.
Gate: trainer changes pass `smoke_v3.py`; teacher output ≥ 10M accepted
tokens/day sustained.

**Phase 1 — step 1, the lever test (week 2–3, 3060, free).**
Base: `checkpoints/brittain2_235m_fim_2k.pt`, 2K context, existing v2 trainer
(the same continuation path used for the 2K stage, fed a plain LM pack with no FIM transform; no doc masking needed at 2K).

| Stage | Tokens | Data | 3060 time |
|---|---:|---|---|
| continued pretrain | 300–500M | 70% verified synthetic (all four families), 20% top-decile real code as replay, 10% prose — plain LM loss, no FIM | ~2–3 days at ~2–3k tok/s |
| instruct SFT | 60M ×2 | the instruction set above, Alpaca template, response-only loss, 20% base replay (`scripts/train/code_sft.py`) | ~3 h |
| RFT ×2 | ~30M/round | prompt the instruct model, k=8 on held-in tasks, keep passers, retrain; then DPO on pass/fail pairs | ~4 h/round + Mac for sampling |

Low LR (1e-4 peak, cosine to 1e-5) — this is a continuation, not a new run.
Watch code BPB on the frozen fixture as the regression alarm, not as the goal.

Gate, measured on the annealed final, same sampling as `MODELS.md`:

| Metric | today | pass |
|---|---:|---:|
| novice_v2 pass@1, base after continued pretrain (completion prompt) | ~1–2% | ≥ 8% |
| HumanEval pass@1, instruct after SFT+RFT (templated) | 2.4% (FIM) | ≥ 8% |
| Syntax validity, instruct | 62% | ≥ 85% |
| Runaway to cap, instruct (greedy) | 18% | ≤ 8% |

Pass → ship it as `brittain2-coder:235m-instruct-v2` and proceed to step 2.
Fail → the synthetic lever does not work at this scale on this hardware; do
not start the 181M. Dump the raw generations before believing any zero.

**Phase 2 — 49M ablations for step 2 (week 3–4, 3060, ~8 h each).** Same 49M
shape as the pilot, ~700M tokens each so results compare to the pilot directly:

| Run | Question |
|---|---|
| A0 | synthetic-dominant mix, AdamW, no doc mask — the baseline |
| A1 | + doc masking |
| A2 | + Muon |
| A3 | synthetic 25% vs 50% |

Optional matched-token control: the best variant at 4B tokens (~2 days) scored
against XS 4B, which is the comparison the pilot could not make.

Gate: best 49M run reaches **novice_v2 pass@1 ≥ 5%** and syntax ≥ 90% at
700M tokens — i.e. at least 5x the pilot on the same shape and token count.

**Phase 3 — 181M pretrain (weeks 4–7, 3060).** Stages A+B. Eval every 500
updates on novice_v2 (20 samples, all 30 tasks — cheap) so the curve is
visible, not just val loss. Gate: base checkpoint ≥ 20% novice_v2 pass@1,
≥ 5% HumanEval before any L4 money is spent.

**Phase 4 — extend + post-train (week 8, L4 ≈ $35).** Stages C–E.
RFT gate: check pass@8 on the held-in set first; < 5% means zero gradient and
the round is dead — fall back to more SFT data instead.

**Phase 5 — release.** Ship the annealed instruct final only (the base stays
on disk as a dev artefact). Score through the template on novice_v2,
HumanEval, runaway, bare-instruction compliance, tool-JSON validity; BPB as a
regression check. Write the `MODELS.md` entry.

## What could go wrong

- **Teacher throughput is lower than assumed.** Then synthetic share falls;
  the anneal still gets all of it. The classifier-filtered real code carries
  the stable phase either way. Measure batched Ollama throughput in week 1.
- **Doc masking costs more than expected on the 3060.** FlexAttention on Ampere
  is slower than the fused SDPA kernel. If the tax is > 20%, use it only in
  stage C on the L4 and pack stage A by file with EOT separators as now.
- **Step 1 or the 49M gate fails.** Stop and diagnose the data, per the gate. This is the
  whole reason the gate exists.
- **Val loss disagrees with the capability evals** — it will. Decide on the
  evals; `MODELS.md` records three runs where `_best` was wrong.
- **12GB at 8K context.** Even with mb 1 and chunked logits it may not fit
  cleanly on the 3060; that is why stage C is on the L4.

## Not in this plan

- A larger vocabulary or a new tokenizer (frozen; IDs baked into checkpoints).
- Logit distillation from the teacher (a 30B forward over billions of tokens is
  not affordable on this hardware).
- Any RL beyond rejection sampling + DPO. Test execution is the reward; no
  reward model.
- Tool-use trajectories before the code gates pass (per `brittain3-plan`).
