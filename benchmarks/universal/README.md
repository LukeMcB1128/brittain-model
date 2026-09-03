# BRITTAIN Universal Generation Benchmark

A multi-track evaluation suite designed to benchmark all models across the BRITTAIN lineage (`BRITTAIN-1`, `BRITTAIN-2`, `BRITTAIN-3`, and `BrittainScript` specialists) under fair, identical conditions.

## Tracks

The suite comprises 75 versioned tasks across five dedicated tracks (15 tasks per track):

| Track | Target Domain | Key Measurements | Toolchain |
|---|---|---|---|
| **Python** | Algorithmic, OOP, parsing, state, math | AST syntax %, Pass@1 %, Solved count, Repetition collapse % | `python3` (built-in `ast` & isolated subprocess) |
| **JavaScript** | ES6+ arrays, objects, strings, closures | Syntax validity via `node --check`, Pass@1 % | `node` |
| **TypeScript** | Typed interfaces, generics, type guards | Type/syntax validity via `tsc --noEmit`, Pass@1 % | `tsc` |
| **JSON** | Configs, schemas, data records, payloads | Strict JSON parse (`json.loads`), Schema key adherence %, Structural closure % | Python `json` |
| **Prose** | Technical, narrative, expository generation | Distinct-1 / Distinct-2 lexical diversity, Sentence closure %, Repetition collapse %, BPB | `nltk`/regex analysis & cross-entropy |

## Ground-Truth Integrity

Every task includes an entry in `reference.jsonl`. Running:

```bash
python3 scripts/evaluate/universal_bench.py --validate
```

validates that all 75 reference solutions parse, compile, type-check, schema-verify, and pass all hidden test harnesses with 0 errors.

## CLI Usage

### Basic Multi-Model Evaluation

```bash
python3 scripts/evaluate/universal_bench.py \
  checkpoints/brittain2_50m_bs.pt \
  checkpoints/brittain2_235m_weights.pt \
  checkpoints/brittain3_49m_pilot/brittain3-xs-coder:49m-pilot.pt
```

### Quick Smoke Test

```bash
python3 scripts/evaluate/universal_bench.py checkpoints/brittain2_50m_bs.pt --quick
```

### Run Specific Tracks

```bash
python3 scripts/evaluate/universal_bench.py checkpoints/brittain2_235m_weights.pt \
  --tracks python,json,prose
```

### Export Results to JSON

```bash
python3 scripts/evaluate/universal_bench.py checkpoints/brittain2_235m_weights.pt \
  --output benchmarks/results/my_eval.json
```
