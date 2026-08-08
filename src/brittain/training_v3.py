"""Configuration, batching, and scheduling helpers for Brittain3 training."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model_v3 import Brittain3Config
from .paths import PROJECT_ROOT


SUPPORTED_CONTEXTS = (1024, 2048, 4096, 8192, 16384)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: str | Path) -> dict[str, Any]:
    with resolve_project_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class StageConfig:
    name: str
    context: int
    microbatch: int
    accumulation: int
    updates: int
    warmup_updates: int
    decay_updates: int
    peak_lr: float
    min_lr: float
    train_data: str
    validation_data: str

    def __post_init__(self):
        if self.context not in SUPPORTED_CONTEXTS:
            raise ValueError(f"context must be one of {SUPPORTED_CONTEXTS}")
        for name in ("microbatch", "accumulation", "updates"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_updates < self.updates:
            raise ValueError("warmup_updates must be in [0, updates)")
        if not 0 <= self.decay_updates <= self.updates - self.warmup_updates:
            raise ValueError("decay_updates does not fit in the stage")
        if not 0 < self.min_lr <= self.peak_lr:
            raise ValueError("stage learning rates are invalid")

    @property
    def tokens_per_update(self) -> int:
        return self.context * self.microbatch * self.accumulation


def parse_training_config(
    training: dict[str, Any], model_cfg: Brittain3Config
) -> list[StageConfig]:
    """Validate one saved training plan and return its context stages."""
    if training.get("format") != "brittain3-training-v1":
        raise ValueError("unsupported Brittain3 training configuration")
    if training.get("precision") not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be bf16, fp16, or fp32")
    if not isinstance(training.get("seed"), int):
        raise ValueError("training seed must be an integer")
    if not training.get("tokenizer_path") or not training.get("output_dir"):
        raise ValueError("tokenizer_path and output_dir must be set")

    optimizer = training.get("optimizer", {})
    fixed_optimizer = {
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
    }
    for name, expected in fixed_optimizer.items():
        if optimizer.get(name) != expected:
            raise ValueError(f"optimizer {name} must be {expected}")

    evaluation = training.get("evaluation", {})
    for name in ("interval", "batches", "plateau_evaluations"):
        if not isinstance(evaluation.get(name), int) or evaluation[name] <= 0:
            raise ValueError(f"evaluation {name} must be a positive integer")
    if evaluation.get("minimum_delta", -1) < 0:
        raise ValueError("evaluation minimum_delta cannot be negative")

    rows = training.get("stages")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training configuration must contain at least one stage")
    stages = [StageConfig(**row) for row in rows]
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("training stage names must be unique")
    if [stage.context for stage in stages] != sorted(stage.context for stage in stages):
        raise ValueError("training contexts must be in increasing order")
    for stage in stages:
        if stage.context > model_cfg.max_seq_len:
            raise ValueError(f"stage {stage.name} exceeds model max_seq_len")
        if not stage.train_data or not stage.validation_data:
            raise ValueError(f"stage {stage.name} must set train and validation data")
    token_batches = {stage.tokens_per_update for stage in stages}
    if len(token_batches) != 1:
        raise ValueError(
            f"all stages must use one effective token batch, got {sorted(token_batches)}"
        )
    return stages


def load_training_config(path: str | Path) -> tuple[dict, Brittain3Config, list[StageConfig]]:
    training = load_json(path)
    if training.get("format") != "brittain3-training-v1":
        raise ValueError("unsupported Brittain3 training configuration")
    model_cfg = Brittain3Config(**load_json(training["model_config"]))
    stages = parse_training_config(training, model_cfg)
    return training, model_cfg, stages


def stage_learning_rate(stage: StageConfig, update: int) -> float:
    """Warmup, stable, and cosine-decay schedule for one stage."""
    if not 0 <= update < stage.updates:
        raise ValueError("stage update is outside the configured range")
    if stage.warmup_updates and update < stage.warmup_updates:
        return stage.peak_lr * (update + 1) / stage.warmup_updates
    decay_start = stage.updates - stage.decay_updates
    if stage.decay_updates == 0 or update < decay_start:
        return stage.peak_lr
    if stage.decay_updates == 1:
        return stage.min_lr
    progress = (update - decay_start) / max(1, stage.decay_updates - 1)
    return stage.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (
        stage.peak_lr - stage.min_lr
    )


class PackedBatchStream:
    """Deterministic shuffled batches with an exact resumable cursor."""

    def __init__(self, path: str | Path, batch_size: int, seed: int):
        loaded = np.load(resolve_project_path(path), mmap_mode="r")
        self.inputs = loaded["input_ids"]
        self.labels = loaded["labels"]
        if self.inputs.shape != self.labels.shape or self.inputs.ndim != 2:
            raise ValueError("packed input_ids and labels must have the same 2D shape")
        if len(self.inputs) < batch_size:
            raise ValueError(
                f"dataset has {len(self.inputs)} rows but batch size is {batch_size}"
            )
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = torch.randperm(len(self.inputs), generator=self.generator)

    @property
    def context(self) -> int:
        return self.inputs.shape[1]

    def _new_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self.order = torch.randperm(len(self.inputs), generator=self.generator)

    def next(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + self.batch_size > len(self.order):
            self._new_epoch()
        indices = self.order[self.cursor:self.cursor + self.batch_size].numpy()
        self.cursor += self.batch_size
        x = torch.from_numpy(np.asarray(self.inputs[indices], dtype=np.int64))
        y = torch.from_numpy(np.asarray(self.labels[indices], dtype=np.int64))
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
            "batch_size": self.batch_size,
            "rows": len(self.inputs),
            "context": self.context,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["batch_size"] != self.batch_size:
            raise ValueError("resume batch size does not match the dataset stream")
        if state["rows"] != len(self.inputs) or state["context"] != self.context:
            raise ValueError("resume data shape does not match the current dataset")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.order = state["order"].clone()
        self.generator.set_state(state["generator_state"])


def synthetic_dataset(path: str | Path, context: int, vocab_size: int, rows: int, seed: int) -> Path:
    """Write a small deterministic dataset for local smoke tests."""
    destination = resolve_project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, vocab_size, size=(rows, context + 1), dtype=np.uint16)
    np.savez(
        destination,
        input_ids=data[:, :-1],
        labels=data[:, 1:].astype(np.int32),
    )
    return destination
