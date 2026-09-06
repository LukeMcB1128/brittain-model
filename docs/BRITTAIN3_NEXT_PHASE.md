# Brittain3 49M — correctness curriculum

This phase has one purpose: test whether targeted, verified training can make
the 49M base model write simple working code. It comes before the 181M run and
before tool-call instruction tuning.

## Pilot decision

The 49M pilot is a conditional pass for the data and architecture recipe. It is
not a pass for coding ability.

- At a matched 512-token window, its final checkpoint gets 1.020 code BPB and
  1.622 prose BPB. This passes the pilot limits and is close to Brittain2 XS 4B.
- It has no empty completions and less repetition than Brittain2 XS 1B.
- It gets 3 correct novice generations out of 360. Brittain2 XS 4B gets 8.
  This sample is too small for a strong relative conclusion.
- Its absolute novice pass@1 is 0.8%. This is not useful coding ability.

Use
`checkpoints/brittain3_49m_pilot/brittain3-xs-coder:49m-pilot.pt` as the
source. It contains the final weights without the old optimizer. Keep `best.pt`
as a fallback, but do not use it as the default: the final weights have slightly
better BPB and novice pass@1.

## Frozen evaluation policy

Do not change `benchmarks/novice/` or its old result. It is the historical pilot
gate. A larger suite must use a new versioned directory. Both suites must be
removed from every new training corpus before packing.

The expanded suite is frozen in `benchmarks/novice_v2/`. It has 30 independent
tasks: 10 each for Python, TypeScript, and JavaScript. Use 20 samples per task.
Results must name the prompt set:

- `general_prompt_collapse`: the five prompts in `compare.py`.
- `novice_suite_collapse`: the executable novice tasks.

These values can differ because the prompts and generation lengths differ.

The suite tests functions, arrays, objects, loops, parsing, state, error
handling, and small multi-step changes. Python solutions run with isolated
Python. JavaScript solutions run with Node.js. TypeScript uses the compiler in
`tools/typescript/node_modules/.bin/tsc`.

Validate the reference solutions before model evaluation:

```bash
python3 scripts/evaluate/novice.py --validate \
  --tasks benchmarks/novice_v2/tasks.jsonl \
  --reference benchmarks/novice_v2/reference.jsonl
```

## Gates before 181M

The curriculum checkpoint must meet all of these conditions:

| Measurement | Required result |
|---|---:|
| Old 36-task novice pass@1 | at least 5% |
| Expanded-suite novice pass@1 | at least 5% |
| Expanded-suite pass@10 | at least 15% |
| Expanded-suite syntax | at least 90% |
| Empty completions | at most 1% |
| Novice-suite repetition collapse | below 10% |
| General-prompt repetition collapse | below 2% |
| Skill coverage | nonzero pass@1 in every core category |
| Code BPB, 512-token matched window | at most 1.050 |
| Prose BPB, 512-token matched window | at most 1.650 |
| HumanEval | repeatable nonzero result |

Five percent pass@1 is still not a useful final coder. It is a scale-up gate: it
shows that correctness training has a measurable effect before money is spent
on 181M. After later instruction tuning, the minimum novice pass@1 target is 15%.

## First continuation probe

Do not start with the old 100M curriculum assumption. First run the 10M-token
probe in `configs/training/brittain3_49m_curriculum_probe.json`.

- Context: 2,048 tokens.
- Tokens per update: 196,608.
- Updates: 64.
- Tensor positions: 12,582,912.
- Expected valid training labels: about 10,028,000 at the measured 79.7% packing efficiency.
- Peak learning rate: 0.00008, with a new optimizer.

The corpus recipe is in
`configs/data/brittain3_49m_curriculum_corpus.json`:

| Content | Tokens | Share |
|---|---:|---:|
| Test code | 4,000,000 | 40% |
| Companion implementation code | 3,000,000 | 30% |
| Documentation | 1,000,000 | 10% |
| Clear English | 800,000 | 8% |
| Brittain tool protocol | 500,000 | 5% |
| Structured data | 500,000 | 5% |
| Verified local teacher solutions | 100,000 | 1% |
| Verified synthetic exercises | 100,000 | 1% |

Test and companion code each target Python 30%, TypeScript 20%, JavaScript 15%,
Rust 12%, C++ 10%, C 8%, and Go 5%. The builder uses deterministic sampling,
normalized-text deduplication, repository caps, secret checks, and both frozen
evaluation suites. Verified teacher replay cannot exceed 5% of the final
corpus.

Do not repeat the current 42-template exercise set until it fills 50M tokens.
Its report shows that a small number of templates produce most of its unique
documents. Repeating that distribution can improve a narrow test without
creating general coding ability.

Teacher seed generation uses three local models in grouped stages:

1. `qwen3.6:35b-a3b` plans diverse task briefs.
2. `qwen3-coder:30b` writes the solution and first tests.
3. `glm-4.7-flash:latest` reviews it and writes independent edge-case tests.

An item is accepted only when its solution passes both test sets. All three
models passed the corrected seven-language compiler bake-off. Qwen Coder was
the fastest at about 71 output tokens per second. The pipeline is
`scripts/prepare/build_teacher_seeds_v3.py`.

The seed builder permits one repair when author code fails. It permits a repair
of reviewer tests only when those tests do not compile. A reviewer assertion
failure stays rejected because it can mean that the task or solution is wrong.
Before output, the builder rejects novice-suite contamination, secret patterns,
exact duplicates, same-language semantic near-duplicates, and code with an
already accepted structural fingerprint.

The builder writes atomic recovery state after each author or reviewer record.
If a run stops, repeat the same command with `--resume` instead of
`--overwrite`. The default recovery path is the output path plus
`.state.json`. The run refuses recovery when a model or generation setting does
not match the saved state.

Run a small seven-language pipeline check before a production seed build:

```bash
python3 -u scripts/prepare/build_teacher_seeds_v3.py \
  --count 7 --balanced-smoke \
  --output /tmp/brittain3-teacher-smoke.jsonl \
  --report /tmp/brittain3-teacher-smoke.report.json \
  --overwrite
```

After each production generation batch, re-run all tests and write the clean
bank. Corpus packing must use `teacher-seeds.clean.jsonl`, not the raw model
output:

```bash
python3 -u scripts/prepare/audit_teacher_seeds_v3.py --overwrite
```

The audit keeps the raw bank and its rejection details for diagnosis.

## Build and start commands

Build the 10M JSONL corpus and check that `failures` is empty:

```bash
python3 -u scripts/prepare/build_curriculum_corpus_v3.py --overwrite
```

Pack it at the probe context:

```bash
python3 -u scripts/prepare/prepare_brittain3.py \
  --input data/generated/brittain3-curriculum/corpus.jsonl \
  --output-dir data/processed/brittain3-curriculum-probe \
  --block-size 2048 --seed 2031
```

Do not train until the corpus and packed-data reports pass validation.

```bash
python3 -u scripts/train/pretrain_v3.py \
  --config configs/training/brittain3_49m_curriculum_probe.json \
  --init-from 'checkpoints/brittain3_49m_pilot/brittain3-xs-coder:49m-pilot.pt' \
  --device auto
```

`--init-from` loads only model weights and starts a new optimizer and schedule.
Use `--resume` only for an interrupted probe checkpoint.

## Compute plan

The pilot processed 707,788,800 tensor positions in 125,516 seconds on the M3
Max. At the same measured rate, 12,582,912 positions take about 37 minutes. The
estimate does not include evaluation or startup. Measure the probe before a
100M run.

Preferred order:

1. Run corpus preparation and evaluation on the M3 Max.
2. Run the 64-update probe on the M3 Max.
3. Run all old and expanded novice gates.
4. Use the RTX 3050 6GB only with a smaller microbatch and larger accumulation.
5. Keep the L4 budget for the 181M run or for a later context-extension stage.

Continue to the 510-update, 100M configuration only if functional scores improve
and BPB does not regress. Do not spend the L4 budget by default.
