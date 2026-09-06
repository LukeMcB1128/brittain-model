"""Train Brittain3 from versioned configuration files."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import random
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows does not provide the Unix resource module.
    resource = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from brittain.checkpoint_v3 import (
    atomic_torch_save,
    checkpoint_payload,
    restore_rng_state,
    validate_checkpoint,
    validate_initialization_checkpoint,
)
from brittain.model_v3 import Brittain3, Brittain3Config
from brittain.training_v3 import (
    PackedBatchStream,
    StageConfig,
    load_training_config,
    parse_training_config,
    resolve_project_path,
    stage_learning_rate,
    synthetic_dataset,
    file_sha256,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/brittain3_49m_pilot.json")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", default=None,
                       help="resume an interrupted run, including optimizer and data cursor")
    start.add_argument("--init-from", default=None,
                       help="start a new stage from model weights with a fresh optimizer")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--max-updates", type=int, default=None, help="safe local run limit")
    parser.add_argument("--smoke", action="store_true", help="use a tiny model and synthetic data")
    parser.add_argument("--smoke-context", type=int, default=1024)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def select_device(requested):
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available")
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device, precision):
    if device.type == "cpu" or precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16" and (device.type == "mps" or torch.cuda.is_bf16_supported()):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def peak_memory(device):
    if device.type == "cuda":
        return {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {"allocated": torch.mps.current_allocated_memory()}
    if resource is None:
        return {"allocated": None, "process_peak_rss": None}
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        maximum_rss *= 1024
    return {"allocated": None, "process_peak_rss": int(maximum_rss)}


@torch.no_grad()
def evaluate(model, stream, batches, device, precision):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = stream.next(device)
        with autocast_context(device, precision):
            _, loss = model(x, y, return_logits=False)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, training, tokenizer, state, data_state, best, weights_only=False):
    payload = checkpoint_payload(
        model,
        optimizer=optimizer,
        scheduler_state={"stage": state["stage"], "stage_update": state["stage_update"]},
        training_state={**state, "best_validation": best},
        data_state=data_state,
        tokenizer=tokenizer,
        training_config=training,
        include_optimizer=not weights_only,
    )
    atomic_torch_save(payload, path)


def move_optimizer_state(optimizer, device):
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.to(device)


def smoke_configuration(args):
    if args.smoke_context not in (1024, 2048, 4096, 8192, 16384):
        raise SystemExit("--smoke-context must be 1024, 2048, 4096, 8192, or 16384")
    cfg = Brittain3Config(
        vocab_size=512,
        max_seq_len=16384,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        intermediate_size=64,
        activation_checkpointing=True,
        logit_chunk_size=128,
    )
    stage = StageConfig(
        name=f"smoke-{args.smoke_context}", context=args.smoke_context,
        microbatch=1, accumulation=1, updates=max(2, args.max_updates or 2),
        warmup_updates=1, decay_updates=1, peak_lr=1e-3, min_lr=1e-4,
        train_data="", validation_data="",
    )
    output = resolve_project_path(args.output_dir or "runs/brittain3_smoke")
    train_path = synthetic_dataset(output / "train.npz", stage.context, cfg.vocab_size, 4, 11)
    validation_path = synthetic_dataset(output / "validation.npz", stage.context, cfg.vocab_size, 4, 22)
    stage = StageConfig(**{**stage.__dict__, "train_data": str(train_path), "validation_data": str(validation_path)})
    training = {
        "format": "brittain3-training-v1", "seed": 1337, "precision": "bf16",
        "compile": False, "output_dir": str(output),
        "tokenizer_path": "synthetic",
        "optimizer": {"betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.1, "gradient_clip": 1.0},
        "evaluation": {"interval": 1, "batches": 1, "plateau_evaluations": 2, "minimum_delta": 0.0},
        "stages": [stage.__dict__],
    }
    return training, cfg, [stage]


def main():
    args = parse_args()
    resume_checkpoint = None
    initialization_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            resolve_project_path(args.resume), map_location="cpu", weights_only=False
        )
        cfg = validate_checkpoint(resume_checkpoint)
        if "optimizer" not in resume_checkpoint:
            raise SystemExit(
                "--resume requires latest.pt or best.pt with optimizer state; "
                "weights.pt is for inference and new training stages"
            )
        training = resume_checkpoint.get("training_config")
        if not isinstance(training, dict):
            raise SystemExit("resume checkpoint does not contain its training configuration")
        stages = parse_training_config(training, cfg)
    else:
        training, cfg, stages = smoke_configuration(args) if args.smoke else load_training_config(args.config)
        if args.init_from:
            if args.smoke:
                raise SystemExit("--init-from cannot be used with --smoke")
            source = resolve_project_path(args.init_from)
            initialization_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            try:
                validate_initialization_checkpoint(
                    initialization_checkpoint, cfg, training["tokenizer_path"]
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            training = copy.deepcopy(training)
            source_state = initialization_checkpoint.get("training_state", {})
            training["initialization"] = {
                "checkpoint": str(source),
                "source_global_update": source_state.get("global_update"),
                "source_tokens_seen": source_state.get("tokens_seen"),
            }
    device = select_device(args.device)
    seed = int(training.get("seed", 1337))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output_dir = resolve_project_path(args.output_dir or training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Brittain3(cfg).to(device)
    source_checkpoint = resume_checkpoint or initialization_checkpoint
    if source_checkpoint:
        model.load_state_dict(source_checkpoint["model"])
    print(f"Brittain3 {model.num_params():,} parameters | device {device} | max context {cfg.max_seq_len}")

    opt_cfg = training["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=stages[0].peak_lr,
        betas=tuple(opt_cfg["betas"]), eps=opt_cfg["epsilon"],
        weight_decay=opt_cfg["weight_decay"],
    )
    if resume_checkpoint and "optimizer" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
    if training.get("compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile failed: {exc}; using eager mode")

    state = {"stage": 0, "stage_update": 0, "global_update": 0, "tokens_seen": 0, "plateau_count": 0}
    best = float("inf")
    if resume_checkpoint:
        state.update(resume_checkpoint["training_state"])
        best = float(state.pop("best_validation"))
        restore_rng_state(resume_checkpoint["rng_state"])

    tokenizer = {
        "name": "brittain3_bpe",
        "path": training["tokenizer_path"],
        "vocab_size": cfg.vocab_size,
        "sha256": (file_sha256(training["tokenizer_path"])
                   if training["tokenizer_path"] != "synthetic" else None),
    }
    started = time.time()
    total_limit = args.max_updates
    completed_this_run = 0
    for stage_index in range(state["stage"], len(stages)):
        stage = stages[stage_index]
        train_stream = PackedBatchStream(stage.train_data, stage.microbatch, seed + stage_index)
        validation_stream = PackedBatchStream(stage.validation_data, stage.microbatch, seed + 1000 + stage_index)
        if train_stream.context != stage.context or validation_stream.context != stage.context:
            raise SystemExit(f"stage {stage.name} data context does not match {stage.context}")
        if resume_checkpoint and stage_index == state["stage"]:
            train_stream.load_state_dict(resume_checkpoint["data_state"]["train"])
            validation_stream.load_state_dict(resume_checkpoint["data_state"]["validation"])
            start_update = state["stage_update"]
        else:
            start_update = 0
        print(f"stage {stage.name}: update {start_update}/{stage.updates}, {stage.tokens_per_update:,} tokens/update")
        model.train()
        for update in range(start_update, stage.updates):
            if total_limit is not None and completed_this_run >= total_limit:
                data_state = {"train": train_stream.state_dict(), "validation": validation_stream.state_dict()}
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_checkpoint(output_dir / "latest.pt", raw_model, optimizer, training, tokenizer, state, data_state, best)
                save_checkpoint(output_dir / "weights.pt", raw_model, optimizer, training, tokenizer, state, data_state, best, weights_only=True)
                print("local update limit reached")
                return
            lr = stage_learning_rate(stage, update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            try:
                for _ in range(stage.accumulation):
                    x, y = train_stream.next(device)
                    with autocast_context(device, training["precision"]):
                        _, loss = model(x, y, return_logits=False)
                        scaled_loss = loss / stage.accumulation
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["gradient_clip"])
                optimizer.step()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise SystemExit(
                        f"out of memory in stage {stage.name}; lower microbatch and raise accumulation"
                    ) from exc
                raise
            state.update({
                "stage": stage_index, "stage_update": update + 1,
                "global_update": state["global_update"] + 1,
                "tokens_seen": state["tokens_seen"] + stage.tokens_per_update,
            })
            completed_this_run += 1
            interval = training["evaluation"]["interval"]
            if (update + 1) % interval == 0 or update + 1 == stage.updates:
                validation = evaluate(
                    model, validation_stream, training["evaluation"]["batches"],
                    device, training["precision"],
                )
                improved = validation < best - training["evaluation"]["minimum_delta"]
                state["plateau_count"] = 0 if improved else state["plateau_count"] + 1
                if improved:
                    best = validation
                data_state = {"train": train_stream.state_dict(), "validation": validation_stream.state_dict()}
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_checkpoint(output_dir / "latest.pt", raw_model, optimizer, training, tokenizer, state, data_state, best)
                if improved:
                    save_checkpoint(output_dir / "best.pt", raw_model, optimizer, training, tokenizer, state, data_state, best)
                save_checkpoint(output_dir / "weights.pt", raw_model, optimizer, training, tokenizer, state, data_state, best, weights_only=True)
                memory = peak_memory(device)
                print(
                    f"stage {stage.name} update {update + 1} | train {float(loss.detach()):.4f} "
                    f"| validation {validation:.4f} | lr {lr:.2e} | memory {memory}"
                )
                if state["plateau_count"] >= training["evaluation"]["plateau_evaluations"]:
                    print(f"PLATEAU: no material validation gain for {state['plateau_count']} evaluations; training continues")
        state["stage"] = stage_index + 1
        state["stage_update"] = 0
        resume_checkpoint = None
    print(json.dumps({
        "status": "complete", "tokens_seen": state["tokens_seen"],
        "best_validation": best, "seconds": time.time() - started,
        "peak_memory": peak_memory(device),
    }, indent=2))


if __name__ == "__main__":
    main()
