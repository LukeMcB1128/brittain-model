# BrittainScript capability benchmark

This suite measures whether a model can produce working BrittainScript, not
merely assign a low loss to existing code.

It has three parts:

1. `tasks.jsonl` contains function headers, hidden test harnesses, and exact
   expected output. The model completes each function and the real
   BrittainScript interpreter runs the result.
2. `translations.jsonl` contains held-out Python programs. The model receives
   the same Python-to-BrittainScript prompt used by the mixed specialist corpus,
   then both programs run and their stdout is compared.
3. `human/` accepts independently written BrittainScript. These files measure
   model BPB on code that did not come through the Python converter.

## Run it

From the repository root:

```bash
python3 scripts/evaluate/bs_capabilities.py \
  checkpoints/brittain_50m_bs.pt \
  checkpoints/xs_bs_native.pt \
  checkpoints/xs_bs_mixed.pt \
  checkpoints/xs_bs_native6.pt \
  --samples 5
```

The default detailed output is `runs/bs_capabilities.jsonl`. A fast plumbing
check is:

```bash
python3 scripts/evaluate/bs_capabilities.py \
  checkpoints/xs_bs_native.pt \
  --samples 1 --max-tokens 40 --translation-max-tokens 60 --skip-human
```

The evaluator expects the BrittainScript repository beside this repository. If
it lives elsewhere, pass `--py2bs-path /absolute/path/to/BrittainScript`.

## Read the metrics

- **Prompted syntax**: the completed program has no interpreter syntax error.
- **Prompted runtime**: it executes without a timeout or interpreter/runtime
  error. A wrong answer can still run.
- **Functional pass@1**: the first fixed-seed completion passes every hidden test
  for that task.
- **Functional pass@k**: at least one of the `k` sampled completions passes every
  hidden test for that task.
- **Functional samples**: fraction of all individual samples that pass.
- **Novel completions**: fraction below the configured 8-gram containment
  threshold against the specialist training split. Very short completions are
  excluded because they do not contain an 8-gram.
- **Translation pass@k**: at least one generated translation exactly matches the
  Python program's stdout.
- **Human BS BPB**: tokenizer-independent compression on the external human set;
  lower is better.

Use at least five samples for model selection. The one-sample command only
checks that the pipeline works.

## Add human-written programs

Put one self-contained `.bs` file per program in `human/`. Do not add converted
corpus examples or anything previously used for training. See
`human/README.md` for the safety and determinism requirements.
