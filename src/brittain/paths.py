"""Stable project paths shared by scripts in this repository."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizers" / "brittain2-code-32k"
BASE_TOKENIZER = TOKENIZER_DIR / "tokenizer.json"
FIM_TOKENIZER = TOKENIZER_DIR / "tokenizer_fim.json"
BRITTAIN3_TOKENIZER_DIR = PROJECT_ROOT / "tokenizers" / "brittain3-code-24k"
BRITTAIN3_TOKENIZER = BRITTAIN3_TOKENIZER_DIR / "tokenizer.json"
BENCHMARK_PROMPTS_DIR = PROJECT_ROOT / "benchmarks" / "prompts"
BS_CORPUS_DIR = PROJECT_ROOT.parent / "bs-corpus"
