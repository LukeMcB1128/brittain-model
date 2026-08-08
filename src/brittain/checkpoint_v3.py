"""Versioned Brittain3 checkpoint save and resume helpers."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model_v3 import (
    BRITTAIN3_ARCHITECTURE,
    BRITTAIN3_ARCHITECTURE_VERSION,
    Brittain3,
    Brittain3Config,
)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    model: Brittain3,
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler_state: dict[str, Any],
    training_state: dict[str, Any],
    data_state: dict[str, Any],
    tokenizer: dict[str, Any],
    training_config: dict[str, Any],
    include_optimizer: bool = True,
) -> dict[str, Any]:
    payload = {
        "architecture": BRITTAIN3_ARCHITECTURE,
        "architecture_version": BRITTAIN3_ARCHITECTURE_VERSION,
        "cfg": model.cfg.to_dict(),
        "model": model.state_dict(),
        "scheduler": scheduler_state,
        "training_state": training_state,
        "data_state": data_state,
        "rng_state": capture_rng_state(),
        "tokenizer": tokenizer["name"],
        "tokenizer_path": tokenizer.get("path"),
        "tokenizer_metadata": tokenizer,
        "training_config": training_config,
        "metadata": model.architecture_metadata(),
    }
    if include_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def validate_checkpoint(checkpoint: dict[str, Any]) -> Brittain3Config:
    if checkpoint.get("architecture") != BRITTAIN3_ARCHITECTURE:
        raise ValueError("this is not a Brittain3 checkpoint")
    if checkpoint.get("architecture_version") != BRITTAIN3_ARCHITECTURE_VERSION:
        raise ValueError("unsupported Brittain3 checkpoint version")
    return Brittain3Config(**checkpoint["cfg"])


def load_brittain3_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
) -> tuple[Brittain3, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = validate_checkpoint(checkpoint)
    model = Brittain3(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return model, checkpoint
