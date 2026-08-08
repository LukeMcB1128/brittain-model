import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.checkpoint_v3 import (
    atomic_torch_save,
    checkpoint_payload,
    load_brittain3_checkpoint,
    restore_rng_state,
)
from brittain.model_v3 import Brittain3, Brittain3Config
from brittain.training_v3 import (
    PackedBatchStream,
    StageConfig,
    load_training_config,
    stage_learning_rate,
    synthetic_dataset,
)


def test_versioned_training_configs_and_effective_batch():
    for path in (
        "configs/training/brittain3_49m_pilot.json",
        "configs/training/brittain3_181m.json",
    ):
        _, model, stages = load_training_config(path)
        assert model.max_seq_len == 16384
        assert len({stage.tokens_per_update for stage in stages}) == 1


def test_warmup_stable_decay_schedule():
    stage = StageConfig(
        "test", 1024, 1, 1, 10, 2, 3, 1e-3, 1e-4, "train", "val"
    )
    assert stage_learning_rate(stage, 0) == 5e-4
    assert stage_learning_rate(stage, 1) == 1e-3
    assert stage_learning_rate(stage, 5) == 1e-3
    assert stage_learning_rate(stage, 9) == 1e-4


def test_batch_stream_exact_resume(tmp_path):
    path = synthetic_dataset(tmp_path / "data.npz", 16, 64, 12, 5)
    first = PackedBatchStream(path, 3, 77)
    first.next(torch.device("cpu"))
    state = first.state_dict()
    expected = first.next(torch.device("cpu"))
    resumed = PackedBatchStream(path, 3, 77)
    resumed.load_state_dict(state)
    actual = resumed.next(torch.device("cpu"))
    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1], actual[1])


def test_checkpoint_restores_model_optimizer_rng_and_metadata(tmp_path):
    random.seed(4); np.random.seed(4); torch.manual_seed(4)
    cfg = Brittain3Config(
        vocab_size=64, max_seq_len=32, n_layer=1, n_head=2,
        n_kv_head=1, n_embd=16, intermediate_size=32,
        activation_checkpointing=False,
    )
    model = Brittain3(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (1, 8)); y = torch.randint(0, 64, (1, 8))
    _, loss = model(x, y, return_logits=False)
    loss.backward(); optimizer.step()
    payload = checkpoint_payload(
        model, optimizer=optimizer, scheduler_state={"stage_update": 1},
        training_state={"global_update": 1, "tokens_seen": 8},
        data_state={"cursor": 1}, tokenizer={"name": "synthetic", "path": "synthetic.json"},
        training_config={"format": "test"},
    )
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(payload, path)
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    loaded, checkpoint = load_brittain3_checkpoint(path)
    restore_rng_state(checkpoint["rng_state"])
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual)
    assert checkpoint["architecture"] == "brittain3"
    assert checkpoint["architecture_version"] == 1
    assert checkpoint["tokenizer"] == "synthetic"
    assert checkpoint["tokenizer_path"] == "synthetic.json"
