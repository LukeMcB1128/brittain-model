Three additions to the BRITTAIN-2 family: a longer-context fill-in-the-middle
model, the first instruction-tuned coder, and a 4x-Chinchilla rerun of the 50M
that BrittainScript trained.

Still from scratch, architecture, tokenizer, data pipeline, training loop,
evaluation and serving are all in this repo.

## New models

| | params | context | tokenizer | what changed |
|---|---|---|---|---|
| **brittain2-coder:235m-fim-2k** | 235M | **2048** | code BPE 32003 | context extension on the FIM model |
| **brittain2-coder:235m-instruct-2k** | 235M | 2048 | code BPE 32003 | code SFT — 32,849 examples |
| **brittain2-xs-coder:50m-bs-4b** | 52M | 512 | code BPE 32000 | 4x Chinchilla, ~4B tokens |

## Measured

Bits per byte over identical frozen held-out text (`data/eval_code.py`,
`english.txt`). Lower is better, and comparable across tokenizers — raw
validation loss is not.

| model | BPB code | BPB prose | syntax valid |
|---|---|---|---|
| 235m base (1K) | 0.751 | 1.259 | 67% |
| 235m fim (1K) | 0.737 | 1.254 | 56% |
| **235m fim-2k** | **0.687** | **1.233** | 62% |
| 235m instruct-2k | 0.851 | 1.554 | — |
| 50m-bs (1B tokens) | 1.080 | 1.702 | 46% |
| **50m-bs-4b (4B tokens)** | **1.002** | **1.603** | 46% |

HumanEval, 10 samples/task, temperature 0.4 / top_p 0.95 / rep 1.12:

| model | pass@1 | pass@10 |
|---|---|---|
| 235m fim (1K) | 2.3% | 6.1% |
| 235m fim-2k | 2.4% | 5.5% |

Runaway generation — completions that hit the token cap instead of stopping,
greedy decoding over 60 tasks:

| model | ran to the cap |
|---|---|
| 235m fim-2k | 58/60 — 97% |
| **235m instruct-2k** | **11/60 — 18%** |

## What each one is actually for

**fim-2k** is the autocomplete model. 2048 tokens is roughly 195 lines of code,
against ~100 before, so it can now see a whole small module rather than the
neighbourhood of your cursor. It reads the code *after* the cursor, not just
before.

**instruct-2k** is the first BRITTAIN model that answers questions about code
rather than continuing it. The SFT was not about raw capability — it was about
knowing when to stop. Runaway generation fell 5.4x. It needs the Alpaca template;
handed a bare instruction it will continue your sentence instead of answering,
which is what `scripts/inference/chat.py` and the server's instruct mode are for.

**50m-bs-4b** is the interesting one. Same 52M model, same corpus, four times the
tokens: BPB code 1.080 -> 1.002. It is still trained by a loop written in
BrittainScript, on a laptop, for nothing.

## Three things worth knowing

**More data helped the small model; more context did not help HumanEval.** The
50M improved 7% on code BPB purely from more tokens. The 2K extension improved
BPB but left HumanEval flat — those prompts are a few hundred tokens and cannot
use a 2048 window. Both results point the same way: **data is the binding
constraint, not parameters and not context.** Throughout the 235M runs train and
val tracked within ~0.01, with no overfitting at 63 tokens/parameter.

**Validation loss picked the wrong checkpoint three times out of three.** The FIM
`_best` ran past the hole 44% of the time against 17% for the annealed final. The
2K `_best` lost every BPB metric to its final. The SFT `_best` truncated 32% of
the time against 18%. Each was a different metric and the same conclusion: finish
the anneal, then decide on the capability you actually want. Every checkpoint here
is a final, not a `_best`.

**BPB cannot judge an instruction tune.** The instruct model scores *worse* on
code BPB (0.851 vs 0.687) because it now models Alpaca-formatted instructions
rather than raw source. Selecting on BPB would systematically favour whichever
checkpoint fine-tuned least — that is, the one that did the least work.

## Using them

```bash
git clone https://github.com/LukeMcB1128/brittain-model
cd brittain-model && pip install -r requirements.txt && pip install -e .

# download the checkpoints from this release into checkpoints/
cd checkpoints && shasum -a 256 -c ../release-assets/SHA256SUMS.brittain-2.1
```

Autocomplete, chat, and raw completion over one Ollama-compatible endpoint:

```bash
python3 scripts/inference/serve.py \
    checkpoints/brittain2_235m_fim_2k.pt=brittain2-coder:235m-fim-2k \
    checkpoints/brittain2_235m_instruct_2k.pt=brittain2-coder:235m-instruct-2k \
    checkpoints/brittain2_50m_bs_4b.pt=brittain2-xs-coder:50m-bs-4b \
    checkpoints/brittain2_235m_weights.pt=brittain2-coder:235m-base-1k
```

Point Continue.dev at `http://localhost:11435` as an Ollama provider. Give exactly
ONE model the `autocomplete` role — Continue silently picks one when several claim
it, and the `model:` field must match the name in `/api/tags`, not the filename.

One-shot from the terminal:

```bash
python3 scripts/inference/sample.py checkpoints/brittain2_235m_fim_2k.pt \
    -p "def binary_search(arr, target):" --suffix "    return best"
python3 scripts/inference/chat.py checkpoints/brittain2_235m_instruct_2k.pt
```

**Tokenizers ship in the repo, not in this release** — `tokenizers/brittain2-code-32k/`.
The 2K and instruct models need `tokenizer_fim.json` (vocab 32003); the 50M needs
`tokenizer.json` (32000). Checkpoints record which they need and the loader refuses
a mismatch rather than emitting quietly wrong text.

## Honest limits

235M and 52M parameters. They write syntactically valid, idiomatically consistent
code that is often semantically wrong. `clamp` comes out right, `fibonacci` does
not. HumanEval pass@1 is 2.4%. Treat every completion as a draft — which is the
autocomplete workflow, and what these were built for.
