"""Generate HumanEval+ samples with a BRITTAIN checkpoint.

EvalPlus cannot load BRITTAIN's custom architecture directly, so this script
only performs generation.  EvalPlus then sanitizes and executes the resulting
JSONL in its own Docker image.

Fast deterministic baseline (one completion per problem):

    python scripts/evaluate/generate_humaneval.py \
        checkpoints/brittain2_235m_weights.pt \
        --greedy --samples 1 \
        --output benchmarks/results/humaneval/brittain2-base.jsonl

Sampled run (enables pass@10):

    python scripts/evaluate/generate_humaneval.py \
        checkpoints/brittain2_235m_weights.pt \
        --samples 10 --resume \
        --output benchmarks/results/humaneval/brittain2-base-10.jsonl

Use ``--limit 1 --max_tokens 32`` for a quick smoke test.  This script never
executes generated code.
"""

import argparse
import codecs
import hashlib
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from evalplus.data import get_human_eval_plus, write_jsonl

from brittain import model_bs
from brittain.model import Brittain, GPTConfig
from brittain.tokenizer import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate EvalPlus-compatible HumanEval+ solutions with BRITTAIN."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1,
                        help="completions per task; 10 enables EvalPlus pass@10")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--greedy", action="store_true",
                        help="deterministic top-k=1 decoding")
    parser.add_argument(
        "--fim_prompt",
        choices=("auto", "never", "always"),
        default="auto",
        help=("wrap prompts as <fim_prefix>prompt<fim_suffix><fim_middle>; "
              "auto enables this for FIM tokenizers"),
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("-r", "--repetition_penalty", type=float, default=1.12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N tasks (for smoke tests)")
    parser.add_argument("--mini", action="store_true",
                        help="use EvalPlus's smaller HumanEval+ mini suite")
    parser.add_argument("--resume", action="store_true",
                        help="append only missing task/sample pairs to an existing output")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max_tokens must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in (0, 1]")
    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty must be positive")
    return args


def select_device(requested):
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but torch.backends.mps.is_available() is false")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_counts(path):
    counts = Counter()
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                counts[row["task_id"]] += 1
            except (json.JSONDecodeError, KeyError) as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return counts


def decode_completion(model, enc, cfg, prompt, args, device, use_fim_prompt):
    if use_fim_prompt:
        prompt_ids = ([enc.fim_prefix] + enc.encode(prompt)
                      + [enc.fim_suffix, enc.fim_middle])
    else:
        prompt_ids = enc.encode(prompt)
    if len(prompt_ids) >= cfg.block_size:
        raise ValueError(
            f"prompt is {len(prompt_ids)} tokens but context is {cfg.block_size}"
        )

    # Keep the complete HumanEval signature/docstring in context.  EvalPlus
    # solutions are short enough that this normally leaves the full 256 tokens.
    max_new_tokens = min(args.max_tokens, cfg.block_size - len(prompt_ids))
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    pieces = []

    stop_ids = {enc.eot}
    for name in ("fim_prefix", "fim_suffix", "fim_middle"):
        token_id = getattr(enc, name, None)
        if token_id is not None:
            stop_ids.add(token_id)

    if args.greedy:
        temperature = 1.0
        top_k = 1
        top_p = None
        repetition_penalty = 1.0
    else:
        temperature = args.temperature
        top_k = args.top_k
        top_p = args.top_p
        repetition_penalty = args.repetition_penalty

    autocast = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type in {"cuda", "mps"} else nullcontext())
    with torch.no_grad(), autocast:
        if isinstance(model, Brittain):
            generated_ids = []
            # Stop GENERATING at the first stop token, not just trimming there
            # afterwards. The truncation loop below already discarded everything
            # past this point, so breaking here is output-identical — it only
            # skips work. It matters: measured over 195 HumanEval completions the
            # mean stops at 147 of the 256 allowed tokens, so decoding the full
            # budget every time costs ~1.7x more than it needs to.
            for token in model.stream(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty):
                token_id = token[0, -1].item()
                if token_id in stop_ids:
                    break
                generated_ids.append(token_id)
        else:
            # The 50M BrittainScript architecture predates the KV-cache stream
            # API. Its generator still returns only model tokens after `idx`.
            output = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=0.0 if args.greedy else temperature,
                top_p=None if args.greedy else top_p,
                repetition_penalty=repetition_penalty,
            )
            generated_ids = output[0, idx.size(1):].tolist()

        for token_id in generated_ids:
            if token_id in stop_ids:
                break
            pieces.append(utf8.decode(enc.token_bytes(token_id)))

    pieces.append(utf8.decode(b"", final=True))
    return "".join(pieces)


def write_metadata(path, args, ck, cfg, enc, device, task_count, use_fim_prompt,
                   architecture):
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "checkpoint_iter": ck.get("iter"),
        "checkpoint_val": ck.get("val"),
        "model": {
            "architecture": architecture,
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
        },
        "tokenizer": enc.name,
        "tokenizer_vocab_size": enc.vocab_size,
        "device": device.type,
        "torch_version": torch.__version__,
        "tasks": task_count,
        "samples_per_task": args.samples,
        "max_tokens": args.max_tokens,
        "greedy": args.greedy,
        "temperature": None if args.greedy else args.temperature,
        "top_p": None if args.greedy else args.top_p,
        "top_k": 1 if args.greedy else args.top_k,
        "repetition_penalty": 1.0 if args.greedy else args.repetition_penalty,
        "seed": args.seed,
        "human_eval_plus_mini": args.mini,
        "fim_prompt": use_fim_prompt,
        "fim_prompt_mode": args.fim_prompt,
    }
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.output.exists() and not args.resume:
        raise SystemExit(
            f"output already exists: {args.output}\n"
            "Use --resume to fill missing samples, or choose a new output path."
        )

    device = select_device(args.device)
    print(f"loading {args.checkpoint} on {device.type} ...", flush=True)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(loaded, dict) and "cfg" in loaded:
        ck = loaded
        cfg = GPTConfig(**ck["cfg"])
        model = Brittain(cfg).to(device).eval()
        model.load_state_dict(ck["model"])
        enc = load_tokenizer(ck)
        architecture = "brittain2"
    else:
        # train_50m.bs saves a bare ModuleList state_dict rather than a wrapper
        # containing cfg/model keys. Reuse the exact reconstruction loader.
        del loaded
        model, enc = model_bs.load(args.checkpoint, device)
        cfg = SimpleNamespace(
            vocab_size=model.vocab,
            block_size=model.block,
            n_layer=model.n_layer,
            n_head=model.n_head,
            n_embd=model.n_embd,
        )
        ck = {}
        architecture = "brittain2-bs"
    has_fim = bool(getattr(enc, "has_fim", False))
    if args.fim_prompt == "always" and not has_fim:
        raise SystemExit("--fim_prompt always requires a tokenizer with FIM sentinels")
    use_fim_prompt = has_fim if args.fim_prompt == "auto" else args.fim_prompt == "always"
    print(
        f"  {model.num_params():,} params | ctx {cfg.block_size} | "
        f"{enc.name} vocab {enc.vocab_size} | {architecture} | "
        f"prompt mode {'FIM (empty suffix)' if use_fim_prompt else 'left-to-right'}",
        flush=True,
    )

    problems = list(get_human_eval_plus(mini=args.mini).items())
    if args.limit is not None:
        problems = problems[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = existing_counts(args.output) if args.resume else Counter()
    generated = 0
    total_needed = sum(max(0, args.samples - counts[task_id])
                       for task_id, _ in problems)
    print(
        f"generating {total_needed} completion(s) for {len(problems)} task(s) "
        f"-> {args.output}",
        flush=True,
    )

    for task_index, (task_id, problem) in enumerate(problems):
        have = counts[task_id]
        if have >= args.samples:
            print(f"  skip {task_id}: already has {have}", flush=True)
            continue
        for sample_index in range(have, args.samples):
            # Per-sample seeds make a resumed run byte-for-byte reproducible.
            sample_seed = args.seed + task_index * 1_000_003 + sample_index
            torch.manual_seed(sample_seed)
            completion = decode_completion(
                model, enc, cfg, problem["prompt"], args, device, use_fim_prompt
            )
            row = {
                "task_id": task_id,
                "solution": problem["prompt"] + completion,
            }
            write_jsonl(str(args.output), [row], append=True)
            generated += 1
            print(
                f"  {generated:>4}/{total_needed} {task_id} "
                f"sample {sample_index + 1}/{args.samples} "
                f"({len(completion)} chars)",
                flush=True,
            )

    metadata_path = write_metadata(
        args.output, args, ck, cfg, enc, device, len(problems), use_fim_prompt,
        architecture
    )
    print(f"done: {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
