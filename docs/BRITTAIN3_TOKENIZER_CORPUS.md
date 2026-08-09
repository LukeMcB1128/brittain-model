# Brittain3 tokenizer corpus

This pipeline builds the tokenizer corpus. It does not build the model-training
corpus. The versioned policy is in
`configs/data/brittain3_tokenizer_corpus.json`.

## Mixture

The target is 2,000,000,000 UTF-8 bytes:

| Category | Bytes | Share |
|---|---:|---:|
| Source code | 1,200,000,000 | 60% |
| Repository documentation | 400,000,000 | 20% |
| FineWeb-Edu English | 200,000,000 | 10% |
| Structured and configuration text | 195,000,000 | 9.75% |
| Brittain tool text | 5,000,000 | 0.25% |

The code quota has separate language quotas. Python receives 25% of code.
TypeScript receives 17.5%, JavaScript 15%, C and C++ 10%, Rust 7.5%, Go 7.5%,
Java and Kotlin 6%, shell languages 4%, and other languages 7.5%.

Grouped quotas also have stream limits. The C and C++ group is 40% C and 60%
C++. The Java and Kotlin group is 75% Java and 25% Kotlin. The shell group is
80% shell and 20% PowerShell. The other group is 35% Ruby, 35% PHP, and 30%
Swift. Structured text is split across JSON, YAML, TOML, HTML, CSS, and SQL.

The tool share is small by design. Structural tokens are reserved before BPE
training. Repeated synthetic tool text must not control ordinary code merges.
The synthetic examples are tokenizer text only. Do not use them as agent SFT
examples.

## Policy

The collector:

- Keeps source, repository, path, category, language, and license provenance.
- Uses the latest resolved dataset commit and records its exact revision.
- Accepts only the licenses in the versioned allowlist.
- Excludes no-license code.
- Excludes dependencies, generated output, minified lines, lock files, binary
  text, and common secret forms.
- Applies exact and whitespace-normalized duplicate removal.
- Limits bytes from one repository.
- Excludes common HumanEval, MBPP, APPS, SWE-bench, and BigCodeBench repository
  names.
- Writes a report with accepted bytes, shortfalls, rejections, source revisions,
  licenses, and a configuration hash.
- Reports exact completion separately. A corpus can pass completion when every
  category and code-language shortfall is no more than 0.1%. This permits
  whole-document quota rounding without hiding a material shortfall.

License detection can be wrong. Keep the source report and review the selected
licenses before a public release. This pipeline does not replace legal review.

## Safe offline test

Install the local development dependencies:

```bash
python3 -m pip install -e '.[dev,corpus]'
```

Run a small local-only test. This command reads the two Brittain repositories.
It does not change the Brittain app and does not use the network:

```bash
python3 scripts/prepare/build_tokenizer_corpus_v3.py \
  --scale 0.00005 \
  --output /tmp/brittain3-tokenizer-dry.jsonl \
  --report /tmp/brittain3-tokenizer-dry.report.json \
  --allow-shortfall --overwrite
```

Remote sources are disabled unless `--allow-remote` is present.

## Remote access

Both configured remote sources use Hugging Face. No AWS key is required.

1. Accept the terms for `bigcode/the-stack-dedup` on its Hugging Face page.
2. Run `hf auth login` on this computer.
3. Install the corpus extra with `python3 -m pip install -e '.[corpus]'`.

The collector streams the `content` field from The Stack. It also keeps the
repository, path, language, and license fields. It accepts a row only when every
reported license is in the conservative allowlist. The Stack is already
near-deduplicated, but the local collector applies its own exact and normalized
duplicate checks as an additional control.

FineWeb-Edu also streams through Hugging Face. The configured revision is
`v1.0.0`. For both sources, the collector resolves and records the exact dataset
commit used by the run.

## Full corpus build

The next command is a large network operation. It can write more than 2GB and
can take hours. Run it only after Hugging Face access succeeds:

```bash
python3 scripts/prepare/build_tokenizer_corpus_v3.py \
  --allow-remote --overwrite
```

Outputs:

- `data/raw/brittain3-tokenizer/corpus.jsonl`
- `data/raw/brittain3-tokenizer/corpus.report.json`

Both paths are ignored by Git. The collector keeps a `.partial` file if a remote
operation fails. It does not silently call an incomplete corpus complete.

If CodeSearchNet is used as a fallback, join its repository license metadata
first. Export normalized JSONL rows with these fields:

```json
{"text":"...","source":"codesearchnet","category":"code","language":"python","repository":"owner/repo","path":"src/app.py","license":"MIT"}
```

Add the file under `normalized_sources` in the versioned configuration. Do not
label all CodeSearchNet rows with one license.

## Held-out tokenizer evaluation corpus

Build a separate 10MB sample with a different seed. Exclude all documents from
the training corpus:

```bash
python3 scripts/prepare/build_tokenizer_corpus_v3.py \
  --allow-remote --scale 0.005 --seed 9001 \
  --exclude-corpus data/raw/brittain3-tokenizer/corpus.jsonl \
  --output data/raw/brittain3-tokenizer/evaluation.jsonl \
  --report data/raw/brittain3-tokenizer/evaluation.report.json \
  --overwrite
```

## Train and evaluate the tokenizer

This command performs local BPE training. It does not use a GPU:

```bash
python3 scripts/prepare/train_tokenizer_v3.py \
  --input data/raw/brittain3-tokenizer/corpus.jsonl \
  --evaluation data/raw/brittain3-tokenizer/evaluation.jsonl \
  --output tokenizers/brittain3-code-24k/tokenizer.json \
  --report tokenizers/brittain3-code-24k/validation.json \
  --compare-brittain2
```

The code regression gate uses the held-out code category when it is present.
The default maximum regression is 3% relative to the Brittain2 tokenizer. Also
review English, structured, and tool-call results before selection.

Do not replace a tokenizer after model pretraining starts. Its token IDs are part
of every checkpoint.
