# Brittain3 base-model foundation

Brittain3 is a separate model architecture. Brittain1 and Brittain2 continue to
use `src/brittain/model.py`. Brittain3 uses `src/brittain/model_v3.py` and an
explicit checkpoint version.

## Production architecture

| Field | Value |
|---|---:|
| Architecture | decoder-only causal Transformer |
| Parameters | 181,529,216 |
| Vocabulary | 24,576 |
| Maximum context | 16,384 |
| Layers | 18 |
| Width | 896 |
| Query heads | 14 |
| KV heads | 7 |
| Head dimension | 64 |
| SwiGLU width | 2,400 |
| Position system | RoPE, theta 100,000 |
| Normalization | pre-norm RMSNorm and per-head QK RMSNorm |
| Attention | grouped-query fused scaled-dot-product attention |
| Dropout | 0 |
| Linear bias | disabled |
| Embedding tie | token embedding and output head share weights |

The model stores seven KV heads in the cache. It expands them only for the
attention kernel. A batch-one bf16 cache at 16K is approximately 252 MiB.

## Configuration fields

The production model configuration is
`configs/models/brittain3_181m.json`. The pilot configuration is
`configs/models/brittain3_49m_pilot.json`.

- `vocab_size`: Number of tokenizer entries. Brittain3 requires 24,576.
- `max_seq_len`: Largest accepted token position. The production value is 16,384.
- `n_layer`: Transformer block count.
- `n_head`: Query-head count.
- `n_kv_head`: Stored key/value-head count. It must divide `n_head`.
- `n_embd`: Residual width. It must divide by `n_head`.
- `intermediate_size`: SwiGLU hidden width.
- `rope_theta`: RoPE frequency base.
- `rms_norm_eps`: RMSNorm stability value.
- `dropout`: Training dropout. The production value is zero.
- `bias`: Linear-bias switch. Brittain3 rejects `true`.
- `qk_norm`: Enables per-head query and key normalization.
- `activation_checkpointing`: Recomputes block activations during backward.
- `logit_chunk_size`: Sequence positions sent to the output head at one time
  when training does not request returned logits.
- `architecture` and `architecture_version`: Checkpoint compatibility keys.

Training files under `configs/training/` define optimizer settings, data paths,
evaluation frequency, and context stages. Each stage uses the same number of
tokens per optimizer update by changing microbatch size.

Training configuration fields are:

- `format`: Version of the training-file format.
- `model_config`: Path to the architecture configuration.
- `tokenizer_path`: Path to the exact tokenizer for the run.
- `output_dir`: Checkpoint destination.
- `seed`: Seed for model, data-order, and random-number state.
- `precision`: `bf16`, `fp16`, or `fp32`. The supplied plans use `bf16`.
- `compile`: Enables `torch.compile` on CUDA. Training continues in eager mode
  if compilation fails.
- `optimizer`: AdamW `betas`, `epsilon`, `weight_decay`, and `gradient_clip`.
  Brittain3 validates the fixed values before it starts.
- `evaluation.interval`: Updates between validation runs.
- `evaluation.batches`: Validation batches in one report.
- `evaluation.plateau_evaluations`: Reports a plateau after this many results
  without a material gain. It does not stop the run.
- `evaluation.minimum_delta`: Smallest validation-loss gain that resets the
  plateau count.
- `stages`: Ordered context stages. Each stage sets `name`, `context`,
  `microbatch`, `accumulation`, `updates`, `warmup_updates`, `decay_updates`,
  `peak_lr`, `min_lr`, `train_data`, and `validation_data`.

A resume operation reads this complete plan from the checkpoint. A different
command-line configuration cannot silently change the remaining stages.

## Tokenizer

Brittain3 uses a new byte-level BPE. It includes EOT, padding, FIM, repository,
file, role, tool-call, tool-result, and message-end tokens from the start.

See `docs/BRITTAIN3_TOKENIZER_CORPUS.md` for the versioned 2GB mixture, license
policy, offline test, remote access steps, and held-out evaluation procedure.

The tokenizer script reads local files only:

```bash
python3 scripts/prepare/train_tokenizer_v3.py \
  --input data/raw/brittain3-tokenizer/corpus.jsonl \
  --jsonl-field text \
  --output tokenizers/brittain3-code-24k/tokenizer.json \
  --evaluation data/raw/brittain3-tokenizer/evaluation.jsonl \
  --compare-brittain2
```

This command reads a prepared corpus and can take time. It does not download
data. Do not use a small smoke corpus as the release tokenizer.

Validate an existing tokenizer:

```bash
python3 scripts/prepare/train_tokenizer_v3.py \
  --validate-only \
  --output tokenizers/brittain3-code-24k/tokenizer.json \
  --compare-brittain2
```

## Local data format

`prepare_brittain3.py` reads one JSON object per line:

```json
{"repository":"owner/project","path":"src/app.py","text":"...","source":"python","is_code":true}
```

It assigns complete repositories to train or validation. It controls the mix by
token counts. It applies FIM after it selects the final document window. It then
packs complete segments without splitting a FIM sequence across rows.

Example for a small local fixture:

```bash
python3 scripts/prepare/prepare_brittain3.py \
  --input data/raw/brittain3-small.jsonl \
  --block-size 1024 \
  --weights '{"python":0.6,"javascript":0.2,"english":0.2}'
```

Preparing the full corpus is a large operation. Do not start it until the pilot
data sources, licenses, decontamination rules, and mixture are approved.

## Tests and smoke runs

Install the development dependency in a local environment, then run:

```bash
python3 -m pytest -q
```

Run a tiny CPU training and checkpoint test:

```bash
python3 scripts/train/pretrain_v3.py \
  --smoke --smoke-context 1024 --max-updates 2 --device cpu
```

Run the real fused 16K forward and backward test on capable hardware:

```bash
BRITTAIN_RUN_REAL_16K=1 python3 -m pytest -q \
  tests/test_brittain3_model.py -k real_fused_16k
```

The default tests also check all five context lengths with a linear test
attention function. That test verifies position, activation, loss, and backward
plumbing without allocating a 16K attention score matrix. The command above is
the separate real-kernel requirement.

Resume a smoke checkpoint:

```bash
python3 scripts/train/pretrain_v3.py \
  --smoke --smoke-context 1024 --max-updates 1 \
  --resume runs/brittain3_smoke/latest.pt
```

## Inference and evaluation

The shared sampler and server detect `architecture: brittain3` in a checkpoint:

```bash
python3 scripts/inference/sample.py \
  checkpoints/brittain3_181m/weights.pt \
  --prompt $'def fibonacci(n: int) -> int:\n'
```

FIM:

```bash
python3 scripts/inference/sample.py \
  checkpoints/brittain3_181m/weights.pt \
  --prompt $'def total(values):\n' \
  --suffix $'    return result\n'
```

Capability smoke report:

```bash
python3 scripts/evaluate/smoke_v3.py \
  checkpoints/brittain3_181m/weights.pt \
  --samples 3 --long-context \
  --output runs/brittain3-smoke-report.json
```

The long-context probe proves that the real 16K computation path uses an early
token. It is not a repository-quality score. A trained checkpoint must also pass
repository and execution evaluations before release.

## Long training commands

The supplied 49M pilot plan uses 707,788,800 tokens. The 181M plan uses
5,200,412,672 tokens:

| Stage | Updates | Tokens |
|---|---:|---:|
| 1K | 15,260 | 4,000,317,440 |
| 2K | 2,670 | 699,924,480 |
| 4K | 1,145 | 300,154,880 |
| 8K | 458 | 120,061,952 |
| 16K | 305 | 79,953,920 |

Do a short L4 throughput check before the main run. Use the measured tokens per
second to calculate the remaining hours and cost. Stop and save the exact resume
checkpoint if the result does not fit the approved credit limit. Do not use all
credits before the 8K and 16K stages have a measured budget.

The following command starts the 49M pilot. It is a long local training run. Do
not run it as a smoke test:

```bash
python3 scripts/train/pretrain_v3.py \
  --config configs/training/brittain3_49m_pilot.json \
  --device cuda
```

The following command starts the paid 181M run. It can consume cloud credits.
Do not run it without separate approval:

```bash
python3 scripts/train/pretrain_v3.py \
  --config configs/training/brittain3_181m.json \
  --device cuda
```

## Hardware sequence

1. Use the M3 Max and RTX 3060 for data preparation, tokenizer work, and the
   49M pilot.
2. Use the L4 for 181M foundation training and long-context adaptation.
3. Use the RTX 3060 for later agent and tool SFT.
4. Use the RTX 3050 6 GB for inference and application tests.

The base-model implementation does not start agent SFT or modify the separate
Brittain app repository.

## Checkpoints and resume

Each resumable checkpoint contains:

- Model and optimizer state.
- Scheduler stage and update.
- Global update and token count.
- Python, NumPy, Torch, and CUDA random-number states.
- Dataset order, epoch, and cursor.
- Best validation result and plateau count.
- Architecture and tokenizer identity.
- The complete training configuration.

Training writes `latest.pt`, `best.pt`, and `weights.pt`. The weights file omits
the optimizer. Writes are atomic.
