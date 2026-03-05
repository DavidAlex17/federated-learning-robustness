from __future__ import annotations

import pytest

from cfg.load_config import validate_and_fill_defaults


def test_fedavg_module_and_config_defaults() -> None:
    pytest.importorskip("flwr")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from experiments.methods.fedavg import run_fedavg

    cfg = validate_and_fill_defaults({})
    assert cfg["method"] in {"smoke_synth", "fedavg"}
    assert cfg["dataset"]["name"] == "mnist"
    assert cfg["dataset"]["train_samples_cap"] > 0
    assert cfg["dataset"]["test_samples_cap"] > 0
    assert callable(run_fedavg)
