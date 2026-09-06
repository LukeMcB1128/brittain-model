# Brittain3 XS-Coder 49M — the pilot

The pilot exists to answer one question before any paid 181M run:

> Does a smaller, quality-first corpus beat a larger, mixed-quality one at the
> same model scale?

Brittain2 XS saw 1B tokens of round-robin Python/JS/TS plus 15% English and still
writes code that is shaped right and wrong underneath. If a 49M Brittain3 trained
on 708M curated tokens cannot beat it, the 181M model will not fix that either,
and the problem is the data or the training recipe rather than scale.

The model configuration is `configs/models/brittain3_49m_pilot.json`
(49,558,592 parameters, 10 layers, width 576, 9 query heads over 3 KV heads).
The schedule is `configs/training/brittain3_49m_pilot.json` and already totals
**707,788,800 tokens** across a 1K stage and a 2K stage.

## Corpus

`configs/data/brittain3_pilot_corpus.json`. Targets are in tokens; byte targets
are derived from the bytes-per-token actually measured on the frozen tokenizer.

| Content | Share | Tokens |
|---|---:|---:|
| High-quality repository code | 60% | 425M |
| Repository documentation and explanations | 15% | 106M |
| Verified novice coding exercises | 10% | 71M |
| General English | 10% | 71M |
| Structured files, patches, and diffs | 5% | 35M |

Within repository code: Python 35%, TypeScript 25%, JavaScript 20%, and the
remaining 20% across Go, Rust, Java, C/C++, and shell.

Two things this corpus is not. It is not the 2GB tokenizer corpus — that one was
built for vocabulary coverage and is frozen and finished. And it is not a
scrape: the exercises category is generated and every accepted solution is
executed before it is kept.

The repository cap is 4MB, down from 20MB for the tokenizer corpus, because 708M
tokens is a third of the size and the old cap would let one large repository own
a visible slice of the pilot's code.

## Build results, 2026-08-09

The remote fetch ran in about ten minutes, not hours.

| Category | Accepted bytes | Shortfall |
|---|---:|---:|
| code | 1,432,024,695 | 0 |
| documentation | 342,002,386 | 0 |
| english | 284,000,445 | 0 |
| structured | 110,051,092 | 0 |
| tool | 3,999,841 | 1,000,159 |

Every code-language quota hit exactly: python 35.0%, typescript 25.0%,
javascript 20.0%, go 5.0%, rust 5.0%, java_kotlin 4.0%, c_cpp 4.0%, shell 2.0%.

The report says `complete: false` **only** because of `tool`, which reached 4MB of
5MB — the Brittain app simply does not contain 5MB of tool-protocol text. That
category is not in the brief; the schema requires it. Nothing the brief asked for
fell short.

## The exercises slice cannot honestly fill 71M tokens

`scripts/prepare/build_exercises_v3.py` generates novice exercises and executes
every one against its own assertions before keeping it. It works — a 60KB smoke
build produced 226 exercises with **zero verification failures** — and it emits
them in the same comment-above-header shape the evaluation uses, so it teaches
the right stopping behaviour rather than the docstring-then-EOT habit.

The template set was expanded from 10 to **42**, covering validation, strings,
collections, parsing, request/response shapes, numbers, classes, and bug-fix
shapes. That was still not enough, and the reason is measurable rather than a
matter of taste. Generating the full 234MB uncapped:

| | Uncapped | Cap 12,000 | Cap 4,000 |
|---|---:|---:|---:|
| Documents | 953,856 | 215,790 | 84,092 |
| Bytes | 234,006,871 | 53,603,292 | 20,621,238 |
| Share from top 10 templates | **75.0%** | 55.6% | 47.6% |
| Templates producing ≥1,000 | 21/42 | 21/42 | 21/42 |

Half the template set cannot produce a thousand unique exercises no matter the
target, because its literal space is small. So 234MB is reachable *only* by
letting about ten list-and-integer templates supply three quarters of it. The
classes, grouping, sorting, and response-shape templates — the parts of the
curriculum the plan actually names — contributed between 3 and 12 documents each
out of 953,856.

**Decision: ship the balanced 53.6MB, not the lopsided 234MB.** This deviates
from the brief's 10% and is recorded here as a deliberate change.

The reason is the pilot's purpose. 71M tokens of roughly ten concepts, in
categories that overlap the novice suite's own categories, is a recipe for the
model to overfit those patterns and post a novice score that does not generalise.
That is a **false positive on the go/no-go gate** — the single worst outcome
available, because it greenlights the paid 181M run on evidence that is not real.
A smaller, genuinely varied exercise slice cannot cause that.

The corpus is ~2.226GB, roughly 665M tokens, against a 707,788,800-token
schedule: about **1.07 epochs**. Nearly everything the model sees is unique,
which was the point of the exercise.

To revisit: raise `--max-per-template`, or add high-entropy templates for the
under-producing concepts, and rebuild. It takes about two minutes.

### Reading the novice score with this in mind

Brittain2 XS never saw a novice curriculum. The pilot will. Some of any
improvement on the novice suite is therefore "we added a curriculum", not
"quality-first data works" — the two are confounded in that one metric. The
entry points and prompts are decontaminated, so this is skill transfer rather
than memorisation, but it is still not a like-for-like comparison.

**BPB and repetition collapse are the clean comparisons.** Treat a novice-suite
gain as corroborating evidence, not as the primary result.

## Decontamination

The novice suite is the pilot's gate, so the corpus must drop any document
containing a suite prompt, a suite test body, or a suite entry-point name in a
definition. Without that the gate measures memorization and the pilot answers the
wrong question. The repository-substring list in the config covers HumanEval,
MBPP, APPS, SWE-bench, BigCodeBench, LiveCodeBench, and EvalPlus.

**Suite decontamination is NOT yet applied.** The collector's `--exclude-corpus`
matches whole documents by exact and whitespace-normalized hash. The novice tasks
are small snippets, so a repository file that merely *contains* one will never
hash-match — the flag gives effectively zero protection here. It needs
entry-point-name and prompt-substring filtering, which the collector does not do.

This is a post-filter on a local JSONL, not a re-download, so it can be applied to
`corpus.jsonl` after the fetch and before packing. **It must be applied before
`prepare_brittain3.py` runs.** Until it is, any novice score from a model trained
on this corpus is untrustworthy.

## Measured baselines

Run on this Mac, 2026-08-09, with the frozen `benchmarks/prompts` samples:

```bash
python3 scripts/evaluate/compare.py \
  checkpoints/brittain2_50m_bs.pt checkpoints/brittain2_50m_bs_4b.pt --samples 100
```

| Checkpoint | Tokens seen | BPB code | BPB prose | Syntax |
|---|---:|---:|---:|---:|
| `brittain2_50m_bs.pt` | ~1B | 1.080 | 1.702 | 53% |
| `brittain2_50m_bs_4b.pt` | ~4B | **1.002** | **1.603** | **38%** |

On the novice suite, 10 samples per task:

| Checkpoint | pass@1 | pass@10 | solved | syntax | empty | collapse |
|---|---:|---:|---:|---:|---:|---:|
| `brittain2_50m_bs.pt` | 0.8% | 2.8% | 1/36 | 82% | 3% | 26.1% |
| `brittain2_50m_bs_4b.pt` | **2.2%** | **5.6%** | **2/36** | **87%** | **1%** | **14.7%** |

**The 4B checkpoint is stronger on every functional axis** — pass@1, syntax,
empty rate, and collapse — as well as on both BPB numbers. The only metric that
favours the 1B checkpoint is `compare.py`'s syntax validity (53% against 38%),
and that measurement is five bare-signature prompts against the novice suite's
36 prompts times 10 samples. Where the two disagree, believe the functional
suite: it executes code, and it is a far larger sample.

This kills a tempting story. It is *not* true that four times the data made the
model worse — that reading came from the narrower metric alone. The 4B run
improved everything that can be measured functionally, including halving
repetition collapse. What remains true is the point that motivates the pilot:
both checkpoints solve at most 2 of 36 novice tasks, so neither is usable, and
4x the tokens moved pass@1 from 0.8% to 2.2%. Scale is working, but far too
slowly to reach a usable coder by that route.

## Targets

**The bar is split**, decided 2026-08-09: the stated BPB targets from the brief,
plus the 4B checkpoint's generation numbers. Since the 4B checkpoint turned out
to be the stronger model almost everywhere, "beat 4B on generation" is the strict
half and it is what stops the pilot passing on a technicality.

| Metric | XS 1B | XS 4B | Pilot target | Source of the bar |
|---|---:|---:|---:|---|
| Code BPB | 1.080 | 1.002 | below **1.050** | brief |
| Prose BPB | 1.702 | 1.603 | below **1.650** | brief |
| Novice pass@1 | 0.8% | 2.2% | above **2.2%** | 4B |
| Novice pass@10 | 2.8% | 5.6% | above **5.6%** | 4B |
| Novice syntax | 82% | 87% | above **87%** | 4B |
| Empty completions | 3% | 1% | at most **1%** | 4B |
| Repetition collapse | 26.1% | 14.7% | **none** | brief |
| HumanEval | 0% | 0% | any repeatable nonzero | brief |

The BPB bars are the brief's and are the *easier* of the two baselines — 1.050
beats the 1B checkpoint but loses to the 4B one. That is a deliberate, recorded
choice, defensible only because every generation bar is taken from the stronger
model. Read the halves together: a pilot that clears 1.050 code BPB while solving
fewer than 2 of 36 novice tasks has not passed.

**A pilot that beats 2.2% pass@1 has done something Brittain2 could not.** Both
baselines solve at most 2 of 36. That is the number to watch.

## Novice-code evaluation

HumanEval is too hard and too noisy at this scale — Brittain2 XS scores a flat
0%, so it cannot rank two checkpoints. `benchmarks/novice/tasks.jsonl` is 36
executable tasks, six each over easy functions, arrays, objects, loops, parsing,
and state updates.

### Prompt shape is not cosmetic

Every prompt is a `#` comment description, a worked example, then a `def` or
`class` header — and it **ends at that header**. This was measured, not assumed.

The suite originally used HumanEval's shape: signature, then a docstring with a
`>>>` example, prompt ending after the closing `"""`. Against
`brittain2_50m_bs` that produced an **empty body in 35 of 36 tasks**. The model
emits EOT immediately, because a closed docstring followed by a newline is a
document boundary in its training data. Isolating the cause, 12 samples each:

| Prompt ends with | Empty completions |
|---|---:|
| `def sum_list(values):` | 0/12 |
| `# comment` line after the signature | 12/12 |
| one-line docstring | 12/12 |
| docstring with a doctest | 12/12 |
| `# comment` **above** the signature | 0/12 |

Anything ending on a completed comment or docstring stops the model dead.
Moving the description above the header fixed it: 0 of 60 empty across six
tasks, with a nonzero pass rate. `tests/test_novice_suite.py` enforces the shape
so it cannot regress.

The lesson generalises past this suite: at this scale a benchmark can report a
flat zero because of where its prompts end, and that is indistinguishable from
the model being incapable unless you look at the raw generations. **Always dump
a few completions before believing a 0%.**

```bash
python3 scripts/evaluate/novice.py checkpoints/brittain2_50m_bs.pt --samples 10
```

Validate the suite itself before trusting a score from it:

```bash
python3 scripts/evaluate/novice.py --validate
```

That runs all 36 reference solutions from `benchmarks/novice/reference.jsonl`. A
suite whose own solutions fail measures its own bugs. `tests/test_novice_suite.py`
enforces the same thing in CI, plus unique ids, the six categories, and a length
ceiling on reference solutions so tasks cannot quietly drift toward HumanEval.

Candidate code is **executed**. Each candidate runs in a subprocess under `-I`
with its working directory set to a fresh empty temporary directory, under a
timeout. That is appropriate for code sampled from our own model. It is not a
sandbox for untrusted third-party code — for that, run the whole script inside
`docker run --rm -m 4g --network none`.

## Packing keeps one window per document

`encode_document` calls `_bounded_text`, which reduces every document to a
**single window of about 1,000 tokens** and discards the remainder. A 300KB
source file contributes roughly 1,000 tokens; the other 99% is dropped. This is
deliberate — it keeps one row from becoming a single long file, and it gives the
packer document diversity — but it means *collected* tokens and *usable* tokens
are very different numbers.

Measured over the assembled corpus, stride-80 sample of 780,544 documents:

| Category | Raw tokens | Usable tokens | Kept |
|---|---:|---:|---:|
| code | 420M | 200M | 48% |
| documentation | 112M | 51M | 45% |
| english | 62M | 39M | 63% |
| exercises | 16M | 16M | 100% |
| structured | 38M | 12M | 32% |
| tool | 3M | 1M | 30% |
| **total** | **651M** | **320M** | **49%** |

At those numbers the 707,788,800-token schedule would have run **2.21 epochs**
over 320M unique tokens — flatly contradicting the brief's "708M mostly unique
tokens", and reproducing in miniature the repetition risk the brief warns about.

The fix is to collect more, not to train longer on less:
`configs/data/brittain3_pilot_corpus_expanded.json` scales the four remote
categories by **2.25x** to 4.88GB, landing usable tokens near 720M and the
schedule near one epoch.

Exercises are **not** scaled with it. Their ceiling is template diversity, not
bytes. Note the pleasant side effect of windowing: exercise documents are short
enough to survive whole, so the 53.6MB balanced slice is 100% kept and lands at
roughly 5% of usable tokens — about twice its raw-byte share, and closer to the
brief's 10% than the byte figure suggests.

**Two lessons worth carrying to the 181M plan.** Never size a corpus in collected
bytes alone; measure what survives windowing. And when a schedule and a corpus
disagree about epochs, the corpus is the thing to change.

## Memory: packing had to be rewritten

`pack_segments` builds `list[list[int]]` for every row and two further
list-of-list copies before converting to numpy, so each token costs roughly 70
bytes as a boxed Python int with a pointer. `prepare_brittain3.py` compounded it
by materialising every `EncodedSegment` before packing. Measured on a 17MB slice
and extrapolated, the pilot corpus projected to a **49GB peak on a 38GB machine**.

`pack_segments_streaming` takes an *iterable* of segments, so the caller passes a
generator and only one segment is ever alive, and appends finished rows into
fixed-size numpy blocks concatenated once at the end. Measured peak on a 200MB
slice is 1.0GB, projecting to about 12.5GB for the full corpus.

It is byte-identical to the original — `tests/test_data_v3_packing.py` asserts
equality across three block sizes, three seeds, and five block-row settings, plus
generator consumption, padding masks, and the oversized-segment guard. A
different-but-plausible packer would corrupt training data in a way no downstream
metric would trace back to the packer.

Per-segment spans are no longer written to metadata by default (`--keep-spans`
restores them): one dict per segment turns the metadata JSON into hundreds of
megabytes, and spans are analysis data that training never reads.

## Measurement notes

**Syntax validity is a binomial proportion.** At 5 prompts by 20 samples, sigma
is about 5 points, so the old default could not separate 53% from 60%. The
default is now 100 samples per prompt, putting sigma near 2.2. The same lesson
already cost a run once, when `eval_batches=20` gave a per-eval sigma of 0.134
and single validation readings were meaningless.

**BPB windows must match.** Every model is scored at its own context by default,
and a longer window eats fewer chunk boundaries. The 49M pilot config declares
`max_seq_len` 16384 but is only trained to 2048, so scoring it at 16K against a
512-context Brittain2 would flatter it on window size alone. `compare.py` now
prints the window it used, warns when windows differ, and takes `--block` to
equalise them. **Score the pilot against Brittain2 XS with `--block 512`.**

**Repetition collapse needs its own metric.** A degenerate loop still parses, and
BPB is measured on held-out text the model never generates, so neither existing
metric catches it. `compare.py` and `novice.py` both report the fraction of
generations whose distinct-4-gram ratio falls below 0.40.

**An empty completion is not valid syntax.** `def f():` with only its header
parses, so the first version of this harness scored ~91% syntax for a model that
had emitted no code whatsoever. `novice.py` now reports an `empty` column and
excludes empties from the syntax numerator. A high empty rate is a signal that
the prompt shape is wrong for the model, not that the model cannot write code.

**Leading tabs are normalised to four spaces.** These models often indent with
tabs. That is valid alone, but the class tasks put a space-indented header above
the generated body, and Python rejects the mixture with `TabError` — scoring a
correct answer as a failure. Only leading whitespace is touched.

## Open decision

The pilot brief names `50m-bs-4b` as the comparison but quotes the **1B**
checkpoint's numbers (1.080 / 1.702). Measurement above confirms those belong to
`brittain2_50m_bs.pt`, not the 4B checkpoint. This matters:

- Against the **1B** checkpoint, the stated targets are a real but modest win.
- Against the **4B** checkpoint, the stated BPB targets are *worse than the
  baseline* — 1.05 loses to 1.002 and 1.65 loses to 1.603. The pilot would pass
  its own gate while losing to Brittain2.

Pick the baseline before the run, not after seeing the result.

One asymmetry to hold either way: the pilot trains on **708M tokens against a 1B
baseline**, so it is deliberately handicapped on volume. A win is therefore
strong evidence for the quality thesis. A loss is ambiguous — it could be the
30% token shortfall rather than the data recipe — so a loss should not by itself
condemn the 181M plan without a matched-token control.
