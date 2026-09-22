# Run 4

The plan for the next adapter, written against measurements rather than
impressions. Every number here came from `evals/eval_defects.py` against
`run3-step-0116`, or from `adapters/run3-train.log`.

## What run 3 actually is

| | |
|---|---|
| rows in the mix | 2,197 (1,859 usable, 338 dropped) |
| target tokens | 316,524 |
| steps | 116 at accum 16 |
| wall clock | 136 minutes |
| peak memory | 11.05 GB |
| learning rate | 1e-4 peak, cosine |
| checkpoints | every 25 steps |
| mix | brittainscript 600, trajectory 600, general 500, restraint 276, identity 221 |

Throughput works out at roughly **2,330 target tokens per minute**, which is
what the estimate in Phase 2 is costed on.

## Baseline, 12 samples per probe

| probe | run 3 |
|---|---|
| settled syntax (`reverse a string`) | reached for a tool it did not need **11/12** |
| exam facts after a course lookup | unsupported figure **6/12**, 1 undecided |
| asked about its own tool use | false account **3/12**, 3 undecided |
| advice after a failed search | 8/12 called another tool; of 4 graded, 2 undecided |
| a fact that moves (mayor) | checked correctly 11/12 |
| pushed back on a wrong answer | checked correctly 12/12 |
| a long listing | not yet measured at baseline |

## What run 4 must not touch

The most valuable part of this plan is the work it removes.

**Fixed in the server. Training for it would be waste, and the data would
fight the fix.**

- repetition loops — `frequency_penalty 0.3`, measured 3/48 to 0/48
- tool storms on an opening greeting — tools are withheld on that turn
- `requestedTool` matching "search the web" inside "why did you search the web"
- a reference note breeding another reference note

**Measured as working. Do not spend data defending these; do not regress
them.** Accepting correction (12/12) and checking a fact that moves (11/12).
The Antigravity review of the served exchanges reported both as broken. My
probes disagree. That has to be settled in Phase 0 before any data is built
for it, or a whole group gets spent fixing something that is not broken.

**Not reproducible. Do not train on a ghost.** The JSON envelope trailer
(0/27) and curriculum domain bleed (0/80 single-turn, multi-turn unconfirmed).

## Phase 0 — close the measurement gaps

Half a day, and it gates everything after it.

1. **Grade the post-tool turns.** Done: `eval_defects.py` now replays a fixed
   tool result and grades the second reply. Fixed results rather than live
   tools, so a change between runs is the model changing and not the internet.
2. **Export the D1 exchanges and grade them with Jev.** Settles the
   correction/mayor discrepancy against real traffic instead of my probes.
   Needs the machine with node:
   `npx wrangler d1 execute brittain-app --remote --json --command "SELECT created_at, model, payload FROM chat_exchanges ORDER BY created_at DESC LIMIT 400"`
3. **Score run 3 checkpoints 25/50/75/100.** Never done. 0116 is being served
   without anyone knowing it is the best one, and run 1's best was step-0100
   rather than its last. If an earlier checkpoint scores better, part of what
   run 4 is meant to fix may just be overtraining.

## Phase 1 — build the data

One to two days, and the real work.

| group | n | the defect it targets |
|---|---|---|
| `known_syntax` | 150 | 11/12 reaching for a tool it did not need |
| `sourced_figures` | 150 | 6/12 unsupported figures |
| `tool_account` | 120 | 3-6/12 false accounts of its own tool use |
| `no_op_turns` | 100 | tool storms where no server gate exists, i.e. Brittain Code |
| `identity_in_code` | 60 | deferred from run 3 |
| `bs_scope` | 60 | the BrittainScript "primary language" overclaim |
| `no_inference_from_paths` | 40 | inferring a name from a home directory |

**About 680 new examples.** Keep run 3's groups: they are why correction
acceptance sits at 12/12. Trim `brittainscript` from 600 to 400 — 27% of the
mix for one niche skill is a lot now that other things need the room.

That lands run 4 near **2,600 rows, roughly 380k target tokens, about 2.7
hours**.

### Two rules for writing the examples

**Show the behaviour, never describe the forbidden one.** Prohibitions
backfire on this model. Measured on the system prompt: "if the user is just
greeting you, DO NOT use tools" took clean greetings from 44/80 to 25/80, and
a positive rephrasing was no better. There is no reason to expect training
data to behave differently from prompt text, so `known_syntax` examples should
answer the question directly and never mention searching.

**`sourced_figures` must be paired.** Half where the figure is absent from the
sources and the reply gives none; half where it is present and gets quoted
exactly. Train only the first half and the result is a model that refuses
numbers. It already hedges and fabricates in the same reply, so the failure
mode is live.

## Phase 2 to 4 — train, score, serve

- **Train** about 2.7 hours unattended, checkpoint every 25 steps.
- **Score every checkpoint**: `python3 evals/eval_defects.py --samples 12
  --model run4-step-NNNN --out run4-NNNN.json`. Eight probes across five
  checkpoints is a few cents of Jev. This is the step run 3 did not have.
- **Ship only if** the three target defects fall *and* correction and
  fact-checking hold at 11-12/12. The suite reports both, so a trade shows up
  instead of hiding.

## Phase 5 — DPO, optional, afterwards

Only on `tool_account` and correction acceptance, with Jev filtering to
high-confidence pairs. After SFT, never before: the mayor failure is not a
preference problem, it is a habit the model was never taught.

**Not on a general quality score.** In the grader calibration, the reply that
invented "100 questions, three hours, 11 percent of takers hit a 5" scored
1.7 on usefulness against 1.3 for the honest reply that gave the real course
number and said the file does not cover exam format. A general quality signal
would train the fabrication in. The defect questions drive the pairs.

## The risk worth watching

`known_syntax` at 150 examples is the largest single intervention, and the
thing it teaches — do not reach for a tool — is one step away from the
behaviour that currently works. The mayor probe is the canary, at 11/12 with
almost no headroom to give. If it drops while `settled syntax` improves, the
group is too blunt and wants splitting by domain rather than growing.
