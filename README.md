# BRITTAIN

Training, evaluation, and inference code for the BRITTAIN model family.

## Repository map

- `src/brittain/` — shared model, tokenizer, prompting, and path code
- `scripts/prepare/` — corpus and tokenizer preparation
- `scripts/train/` — pretraining, FIM, SFT, and specialist training
- `scripts/evaluate/` — comparable BPB, syntax, and BrittainScript evaluation
- `scripts/inference/` — sampling, chat, and local serving
- `brittain_script/` — training programs written in BrittainScript
- `tokenizers/` — versioned tokenizer assets
- `benchmarks/` — frozen evaluation fixtures and results
- `docs/` — model history and operational notes
- `data/` — local/generated datasets; large contents are ignored
- `checkpoints/` — local model weights; ignored by extension
- `runs/` — logs and temporary run artifacts; ignored

## Common commands

```bash
python3 scripts/inference/sample.py checkpoints/brittain_235m_weights.pt -p "def fibonacci(n):\n"
python3 scripts/evaluate/compare.py checkpoints/brittain_124m_best.pt checkpoints/brittain_235m_weights.pt
python3 scripts/evaluate/bs_capabilities.py checkpoints/xs_bs_native.pt --samples 5
python3 scripts/prepare/filter_bs.py --stats
python3 scripts/train/fim.py --help
```

Commands can be launched from any working directory; repository-owned default
paths are resolved from the source tree rather than the shell's current directory.

See [`docs/MODELS.md`](docs/MODELS.md) for model lineage and benchmark history.
See [`benchmarks/brittainscript/README.md`](benchmarks/brittainscript/README.md)
for the executable BrittainScript capability suite.
