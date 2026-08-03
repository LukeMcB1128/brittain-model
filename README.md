# BRITTAIN

Training, evaluation, and inference code for the BRITTAIN model family.

BRITTAIN is a project I started to learn the full language-model pipeline by
building it myself: the architecture, tokenizer, corpus preparation, pretraining,
specialization, evaluation, and inference stack. The models are small on purpose.
They are code-completion models and experiments, not replacements for modern
assistants.

## First BRITTAIN-2 release

| Model | Parameters | Context | What it is |
|---|---:|---:|---|
| `brittain2-xs-coder:50m-bs` | 51,917,824 | 512 | The proof-of-concept model trained by code written entirely in BrittainScript |
| `brittain2-xs-brittainscript-specialist:50m-bs` | 51,917,824 | 512 | The mixed BrittainScript specialist continued from the XS coder |
| `brittain2-coder:235m-base-1k` | 235,176,960 | 1,024 | The base Python, JavaScript, and TypeScript completion model |
| `brittain2-coder:235m-fim-1k` | 235,180,032 | 1,024 | The base coder continued with fill-in-the-middle training |

The complete release notes and benchmark methodology are in
[`brittain2-first-models-notes.md`](brittain2-first-models-notes.md).

## Install

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the checkpoint files from the GitHub release and place them in
`checkpoints/`. The two tokenizer files needed by the release are already under
`tokenizers/brittain2-code-32k/`.

## Run a completion

The same sampler now loads all four release checkpoints.

```bash
python3 scripts/inference/sample.py \
  checkpoints/brittain2_235m_weights.pt \
  -p $'def fibonacci(n: int) -> int:\n'
```

The XS models use the same command:

```bash
python3 scripts/inference/sample.py \
  checkpoints/brittain2_50m_bs.pt \
  -p $'func add(a, b):\n'

python3 scripts/inference/sample.py \
  checkpoints/brittain2_xs_bs_mixed.pt \
  -p $'func add(a, b):\n'
```

For FIM, the prompt is everything before the cursor and `--suffix` is everything
after it. The sampler adds the correct BRITTAIN sentinels itself.

```bash
python3 scripts/inference/sample.py \
  checkpoints/brittain2_235m_fim.pt \
  -p $'def total(xs):\n' \
  --suffix $'    return result\n' \
  --stop_blank
```

These are base models, so incomplete code works better than questions or chat
prompts. The default sampling settings are deliberately cold because code has much
lower entropy than prose.

## Ollama-compatible local server

Install the server dependencies:

```bash
pip install -e '.[serve]'
```

Then load any combination of checkpoints:

```bash
python3 scripts/inference/serve.py \
  checkpoints/brittain2_50m_bs.pt=brittain2-xs-coder:50m-bs \
  checkpoints/brittain2_xs_bs_mixed.pt=brittain2-xs-brittainscript-specialist:50m-bs \
  checkpoints/brittain2_235m_weights.pt=brittain2-coder:235m-base-1k \
  checkpoints/brittain2_235m_fim.pt=brittain2-coder:235m-fim-1k
```

The server runs at `http://localhost:11435`. It exposes `/api/tags`, `/api/show`,
`/api/generate`, and `/api/chat`. `/api/show` reports the real maximum context for
the selected checkpoint. The FIM model also translates the common StarCoder,
CodeLlama, Qwen, and DeepSeek marker formats into BRITTAIN's sentinel format.

## Repository map

- `src/brittain/` — shared model, tokenizer, prompting, and path code
- `scripts/prepare/` — corpus and tokenizer preparation
- `scripts/train/` — pretraining, FIM, SFT, and specialist training
- `scripts/evaluate/` — comparable BPB, HumanEval, syntax, and BrittainScript evaluation
- `scripts/inference/` — sampling, chat, and local serving
- `brittain_script/` — training programs written in BrittainScript
- `tokenizers/` — versioned tokenizer assets
- `benchmarks/` — frozen evaluation fixtures and saved results
- `docs/` — model history and operational notes
- `data/` — local/generated datasets; large contents are ignored
- `checkpoints/` — local model weights; ignored by extension
- `runs/` — logs and temporary run artifacts; ignored

See [`docs/MODELS.md`](docs/MODELS.md) for the complete lineage and architecture
history. See [`benchmarks/brittainscript/README.md`](benchmarks/brittainscript/README.md)
for the executable BrittainScript capability suite.

## License

The original repository code, documentation, tokenizer assets, and released
BRITTAIN model weights are available under the
[Apache License 2.0](LICENSE). Third-party training datasets are not included and
remain under their own licenses and terms. See [NOTICE](NOTICE) for the exact
scope.
