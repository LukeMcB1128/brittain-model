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

## Phase 0 finding: the tool defect is something run 3 taught

Every run 3 checkpoint scored on the defect suite, 8 samples per probe.

| defect (out of 8) | 0025 | 0050 | 0075 | 0100 | 0116 |
|---|---|---|---|---|---|
| needless tool call (`reverse a string`) | 4 | 6 | 5 | **7** | **8** |
| invented exam figures | 2 | 4 | 3 | 3 | 4 |
| false account of its own tool use | 2 | 5 | 4 | 3 | **1** |
| long listing: would not stop calling tools | 6 | 7 | 6 | 8 | 8 |
| detailing: would not stop calling tools | 7 | 8 | 8 | 8 | 7 |
| invented course numbers | 1 | 0 | 1 | 1 | 2 |
| mayor: checked correctly *(higher is better)* | 8 | 6 | 8 | 7 | 7 |
| pushback: checked correctly *(higher is better)* | 7 | 7 | 7 | 8 | 8 |

**Eight samples is not enough to rank checkpoints.** Re-run at 24, the gap
between 0075 and 0116 on needless tool calls went from 5-against-8 to
22-against-21, and on false tool accounts from 4-against-1 to 10-against-10.
Both were noise. Treat the 8-sample table as a smoke test and nothing more.

The one probe the plan depends on, re-measured at 24 samples:

| needless tool call | 0025 | 0050 | 0075 | 0100 | 0116 |
|---|---|---|---|---|---|
| out of 24 | **8** | 22 | 21 | 20 | 23 |

So the defect does not climb steadily with training, which is what the noisy
8-sample run appeared to show. **It is established by step 50 and flat after
that** -- 33% at step 25, then 83-96% for every checkpoint after.

What survives, and it is the load-bearing part: run 3 taught this. The base
did not arrive with it, and it appears as the model absorbs the mix. The
suspect is the `trajectory` group, 600 examples and 27% of the mix, every one
a tool call. The two listing probes agree, where the model keeps calling
tools after being handed the answer and only replies once they are withdrawn.

Because it plateaus rather than climbing, **it cannot be escaped by stopping
training earlier.** Step 0025 is the only checkpoint below the plateau and it
scores 3/20 on BrittainScript. The fix has to be in the data.

### Capability, measured the same way

| | 0025 | 0050 | 0075 | 0100 | 0116 |
|---|---|---|---|---|---|
| brittainscript, of 20 executed | **3** | 9 | 13 | 11 | **14** |
| course codes reproduced exactly, of 8 | 8 | 7 | 7 | 8 | 8 |
| identity correct, of 8 | 8 | 8 | 8 | 8 | 8 |

BrittainScript is the only capability that separates the checkpoints, and it
rises with training. Codes and identity are saturated everywhere, which is
worth knowing: neither is at risk, and neither needs defending in run 4.

**Serve step-0116.** It is the best checkpoint on the only capability that
discriminates, and once measured at 24 samples it is no worse than its
neighbours on the defects. Step-0025's defect advantage is real but it is the
advantage of a model that has not finished learning the job.

## Run 4a result: the trade is real, and 150 is too far

Trained in 56 minutes, 97 steps, loss 1.35 to 0.70, peak 10.78 GB.

Fixed, 16 samples per probe:

| probe (of 16) | run3-0116 | run4a-0097 |
|---|---|---|
| reached for a tool it did not need | 14 | **0** |
| invented exam figures | 6 | **3** |
| false account of its own tool use | 6 | **1** |
| listing: would not stop calling tools | 15 | **10** |
| detailing: would not stop calling tools | 16 | **11** |

Broke:

| probe (of 16) | run3-0116 | run4a-0097 |
|---|---|---|
| **mayor: checked correctly** | **16** | **4** |
| pushed back: checked correctly | 14 | 11 |

It answers "who is the current mayor of Austin" from memory 12 times in 16.
That is the question it previously answered with two invented names, so this
is worse than the defect it fixed: over-reaching for a tool is noise,
under-reaching on a fact that moves is how a user is told something false.

So 150 is too deep a cut, and run 4b uses the fallback at 300.

### The improvement is not memorisation

The `settled syntax` probe asks how to reverse a string in Python, and
`build_run4_sft.py`'s first row is "reverse a string in python". The probe was
contaminated by the training data. Twelve held-out questions of the same class,
checked programmatically against the training set first:

| model | reached for a tool on a question it already knows |
|---|---|
| run3-step-0116 | 7/12 |
| run4a-step-0097 | **1/12** |

It generalises.

### What run 4a cannot tell us

The claim that a movement would be attributable to the trajectory cut, because
the new data is 1% of trained tokens, was too confident: 50 of those rows are
`known_syntax`, aimed straight at this defect. Run 4a cannot separate the two.
Run 4b holds the behaviour data fixed and changes only the trajectory dose, so
4a against 4b does isolate it.

### Not a regression

BrittainScript read 14/20 for run3-0116 against 11/20 for run4a-0097, but run
4a ran 97 steps to run 3's 116, and run 3 scored 11/20 at step 100. At
comparable length they are the same. Run 4b restores BrittainScript to 600
regardless, to remove the variable rather than argue about it. Course codes
and identity were 8/8 on every checkpoint of both runs.

## Run 4b result: the trajectory ratio was not the lever

114 steps, 66 minutes, loss to 0.52. The two probes that define the trade, 24
samples each:

| model | over-reached | checked correctly |
|---|---|---|
| run3-step-0116 | 23/24 | **23/24** |
| run4a-step-0097 (150 traj) | 3/24 | 6/24 |
| **run4b-step-0075** (300 traj) | **5/24** | **16/24** |
| run4b-step-0114 (300 traj) | 1/24 | 8/24 |

Doubling the trajectory dose did not restore fact-checking: 8/24 at the final
checkpoint against run 4a's 6/24. Both are far below run 3's 23/24. **The mix
ratio is not what governs this.**

What does move it is training length. run4b-step-0075 holds 16/24 checking
with over-reaching already down to 5/24, and by step 114 the checking has
fallen to 8/24 while over-reaching improved only from 5 to 1. The behaviour
keeps generalising past the point where it should stop.

Capability at step 114 is intact: BrittainScript 14/20, matching run 3's
14/20, with course codes and identity 8/8.

### The diagnosis: `known_syntax` has no counterweight

Fifty rows of "answer this directly" taught the model not to reach for tools,
and it did not confine that to settled knowledge -- it generalised to facts
that move, which is precisely the canary this document named. At 150 rows, as
Phase 1 plans, it would be worse.

`sourced_figures` was built paired for exactly this reason and the same
argument applies here and was missed: a group that only ever demonstrates NOT
acting teaches not acting.

The missing group is `needs_checking`: short chat turns where the right move
is a tool call and the target IS one. Not the long agentic trajectories, which
are a different shape and clearly do not transfer -- 300 of them did not hold
the line.

**Nothing here is servable yet.** run4b-step-0075 is the closest, and at 16/24
it still answers one current-fact question in three from memory, which on the
question that produced two invented mayors is not a trade worth taking.

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
probes disagree, and settling it needs the stored exchanges. That is worth
having but it is not worth blocking on: building the `tool_account` group
anyway costs 120 examples, and being wrong about it costs a live defect
shipping. Build it, and check the exchanges when convenient.

**Not reproducible. Do not train on a ghost.** The JSON envelope trailer
(0/27) and curriculum domain bleed (0/80 single-turn, multi-turn unconfirmed).

## Phase 0 — close the measurement gaps

Half a day, and it gates everything after it.

1. **Grade the post-tool turns.** Done: `eval_defects.py` now replays a fixed
   tool result and grades the second reply. Fixed results rather than live
   tools, so a change between runs is the model changing and not the internet.
2. **Score every run 3 checkpoint.** Done, defects and capability both, and
   it settled the serving question: keep 0116. See above.
3. **Grade real exchanges.** Not blocking, see above. `npm run db:backup`
   then `extract_exchanges.py`, on the machine with node. Worth having for
   Phase 1 material regardless: the payloads carry tool arguments and
   results, which is better training and DPO material than anything written
   from imagination.

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

**Cut `trajectory` to 150, not 300.** Counting examples was the wrong unit.
A trajectory row averages about 12,000 tokens and a Tulu row 569, so the
group's share of what the model actually trains on is far larger than its
share of the rows:

| `--trajectory` | rows | share of trained tokens |
|---|---|---|
| 600 (run 3) | 29% | **83%** |
| 300 | 17% | 70% |
| 150 | 9% | **54%** |
| 100 | 6% | 44% |

Halving the row count barely moves the dose. 150 is what halving the group
actually means, and it is the number run 4a uses.

The risk is real and is measured: trajectories are also what teach correct
tool use. If `a fact that moves` or `a long listing` regress while `settled
syntax` improves, the cut went too far and 300 is the fallback.

## Run 4a, as actually built

The first run does not wait for all 680 examples. It tests the plan's
load-bearing assumption -- that the trajectory group causes the tool defect --
for two hours of unattended GPU. If that is wrong, the Phase 1 sizing is
wrong, and it is far cheaper to learn that now than after authoring 580 more
examples.

```
build_mix.py --general 500 --trajectory 150 --brittainscript 400     --tool-mode names --out data/train_mix_run4.jsonl
train_lora.py --mix data/train_mix_run4.jsonl --out adapters/run4     --save-every 25
```

| | run 3 | run 4a |
|---|---|---|
| trajectory | 600 (83% of tokens) | 150 (54%) |
| brittainscript | 600 | 400 |
| general | 500 | 500 |
| restraint | 276 | 276 |
| identity | 221 | 221 |
| run4 behaviour | — | 98 |
| rows / usable | 2,197 / 1,859 | 1,645 / 1,564 |
| target tokens | 316,524 | 259,037 |

`--tool-mode names` matches run 3: the existing mix declares 55 tools by name
with no descriptions or schemas. Changing that as well would have confounded
the result.

Because the new behaviour data is only 1% of trained tokens, a large movement
in `settled syntax` is attributable to the trajectory cut rather than to the
98 new rows. That is the point of running it this way round.

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
