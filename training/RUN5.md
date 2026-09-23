# Run 5

Run 4 answered its question: reaching for tools can be tuned, it is one dial,
and a restraint group needs a paired counterweight or it teaches the model to
stop checking. Run 5 has a different job. **Keep run 4c's gains and bring
general capability back to the base.**

## Why: the standing bench, run on adapters for the first time

`run_general_eval.py` and `run_bs_eval.py` could only load a model offline,
which meant no LoRA path, so they had only ever scored the base. They now take
`--server NAME` and send the identical prompt to `/v1/completions`. The base
re-run through that path agrees with its offline record on 493 of 494 items.

| | base | run3-0116 *(served)* | run4c-0116 |
|---|---|---|---|
| ARC | 93.8 | 95.0 | 95.0 |
| GSM8K | 89.0 | 88.0 | 85.0 |
| **HumanEval** | **80.0** | **62.5** | **65.0** |
| MMLU | 76.3 | 73.7 | 75.4 |
| **XSTest safe** *(answers harmless prompts)* | **92.0** | **50.0** | **44.0** |
| XSTest unsafe *(refuses harmful ones)* | 88.3 | 98.3 | 96.7 |
| **capability overall** | **84.7** | 81.7 | 81.7 |
| BrittainScript, no spec, matches | 0.0 | 71.7 | 62.5 |
| tool calls, `names_desc`, args ok of 103 | — | 36 | 38 |
| tool calls, `names`, args ok of 103 | — | 36 | 28 |

`names_desc` is the shape Brittain Code sends at 32k, and it holds. The
`names` drop is plausibly the halved trajectory group baking in fewer
schemas; watch it, it is not the production path.

### Over-refusal is real, and has a suspect

The served model refuses harmless prompts the base answered: 42 of them for
run 3, 48 for run 4c. The replies were read, not inferred from the phrase
matcher -- *"I'm sorry, but I can't provide information on how to exterminate
rats."* That is the Tulu refusal register, and 93 of the 500 general rows,
19%, are Tulu safety data:

| source | rows |
|---|---|
| `tulu_v3.9_wildjailbreak_decontaminated_50k` | 46 |
| `tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k` | 39 |
| `coconot_converted` | 8 |

### HumanEval is a real loss with no cause yet

Ruled out, each by looking rather than assuming:

- **format** -- every failure is properly fenced Python
- **BrittainScript leaking into Python** -- no `push(`, `space(` or `pyimport` in any failure
- **truncation** -- the adapters hit the 640-token limit on 0 of 40 problems,
  and write *shorter* replies than the base (median 173 tokens against 207)

It appears by step 25 in every mix and does not follow any dose:

| adapter | trajectories | HumanEval |
|---|---|---|
| base | -- | 80.0 |
| run3, steps 25/50/75/100/116 | 600 | 65 / 70 / 65 / 65 / 62.5 |
| run4a-0097 | 150 | 72.5 |
| run4b-0114 | 300 | 75.0 |
| run4c-0116 | 300 + 30 needs_checking | 65.0 |

4b and 4c differ only by thirty rows and sit ten points apart, which says 40
problems is below the noise floor for ranking mixes. Attributing this needs a
code eval with power: the full 164-problem HumanEval plus MBPP, about 600
problems.

## Changes, each with the probe that checks it

| change | checked by |
|---|---|
| drop the 93 safety rows from `general` | XSTest safe, and XSTest unsafe, which may fall back to the base's 88 |
| `known_syntax : needs_checking` from 50:30 to 50:15 | over-reach and mayor checking, 24 samples |
| `sourced_figures` from 20 to ~150, paired | invented exam figures, 24 samples |
| HumanEval: undecided until the powered code eval says what | HumanEval |

## The gate, written before training

Run 5 ships only if every line holds. Four runs of borderline numbers being
reread as good enough is the reason these are fixed in advance.

| metric | base | must reach |
|---|---|---|
| capability overall | 84.7 | **>= 84** |
| HumanEval | 80.0 | **>= 75** |
| XSTest safe | 92.0 | **>= 85** |
| XSTest unsafe | 88.3 | **>= 85** |
| BrittainScript, no spec | 0.0 | **>= 65** |
| over-reach, of 24 | -- | **<= 8** |
| mayor checked, of 24 | -- | **>= 21** |
