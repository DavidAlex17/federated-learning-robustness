from __future__ import annotations

import pytest

from cfg.load_config import validate_and_fill_defaults


def test_fedavg_tiny_module_and_config_defaults() -> None:
    pytest.importorskip("flwr")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from experiments.methods.fedavg_tiny import run_fedavg_tiny

    cfg = validate_and_fill_defaults({})
    assert cfg["method"] in {"smoke_synth", "fedavg_tiny"}
    assert cfg["dataset"]["name"] == "mnist"
    assert cfg["dataset"]["train_samples_cap"] > 0
    assert cfg["dataset"]["test_samples_cap"] > 0
    assert callable(run_fedavg_tiny)
