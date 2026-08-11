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

Use `checkpoints/brittain3_49m_pilot/weights.pt` as the source. It contains the
final weights. Keep `best.pt` as a fallback, but do not use it as the default:
the final weights have slightly better BPB and novice pass@1.

## Frozen evaluation policy

Do not change `benchmarks/novice/` or its old result. It is the historical pilot
gate. A larger suite must use a new versioned directory. Both suites must be
removed from every new training corpus before packing.

The next suite will use 20 samples per task. It will add many independent tasks
instead of only adding more samples to the same 36 tasks. Results must name the
prompt set:

- `general_prompt_collapse`: the five prompts in `compare.py`.
- `novice_suite_collapse`: the executable novice tasks.

These values can differ because the prompts and generation lengths differ.

The default scope for the new suite is Python, TypeScript, and JavaScript. Its
core section will test functions, arrays, objects, loops, parsing, state, error
handling, and small multi-step changes. Python solutions run with isolated
Python. JavaScript solutions run with Node.js. TypeScript needs a fixed compiler
version before its scores can become a release gate.

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

## First continuation run

The fixed training plan is in
`configs/training/brittain3_49m_curriculum.json`.

- Context: 2,048 tokens.
- Tokens per update: 196,608.
- Updates: 510.
- Total tokens: 100,270,080.
- Peak learning rate: 0.00008, with a new optimizer.

The first 100M-token corpus target is:

| Content | Share |
|---|---:|
| Diverse, execution-verified solutions | 50% |
| High-quality repository code replay | 25% |
| Verified bug fixes and diffs | 10% |
| Code documentation and clear English | 10% |
| Brittain tool protocol and structured data | 5% |

The verified slice targets Python 30%, TypeScript 20%, JavaScript 15%, Rust 12%,
C++ 10%, C 8%, and Go 5%. Every solution must pass generated tests. Each
semantic task family has a strict cap.
Near-duplicates are removed by normalized syntax and token fingerprints. The
corpus must also pass secret, license, and evaluation-contamination checks.

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

Run a small seven-language pipeline check before a production seed build:

```bash
python3 -u scripts/prepare/build_teacher_seeds_v3.py \
  --count 7 --balanced-smoke \
  --output /tmp/brittain3-teacher-smoke.jsonl \
  --report /tmp/brittain3-teacher-smoke.report.json \
  --overwrite
```

## Start command

Do not run this command until the curriculum NPZ files exist and their report
passes validation.

```bash
python3 -u scripts/train/pretrain_v3.py \
  --config configs/training/brittain3_49m_curriculum.json \
  --init-from checkpoints/brittain3_49m_pilot/weights.pt \
  --device auto
```

`--init-from` loads only the model weights and starts a new schedule. `--resume`
is only for an interrupted curriculum checkpoint.

## Compute plan

The pilot processed 707,788,800 tokens in 125,516 seconds on the M3 Max. At the
same measured rate, 100,270,080 tokens take about 4.9 hours. Use a short run to
measure the new corpus before planning the full run.

Preferred order:

1. Run corpus preparation and evaluation on the M3 Max.
2. Run a 10-update training test on the RTX 3060 12GB and the M3 Max.
3. Use the faster stable device for the 510-update run.
4. Use the RTX 3050 6GB only with a smaller microbatch and larger accumulation.
5. Keep the L4 budget for the 181M run or for a later context-extension stage.

Stop after 100M tokens and run all gates. Add a second 100M-token block only if
functional scores improve and BPB does not regress. Do not spend the full L4
budget by default.
