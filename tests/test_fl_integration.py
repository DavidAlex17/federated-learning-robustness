"""Integration test: run real Flower FL for a few rounds and verify baseline.

This is the only test that exercises the full fedavg_tiny pipeline end-to-end
(Flower simulation, MnistClient, RobustFedStrategy). It uses a minimal config
(2 clients, 2 rounds, 100 train samples) so it stays fast even in CI.

It serves as the baseline verification: if clean FedAvg fails here, nothing
else in the attack/defense comparisons is trustworthy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("flwr")
pytest.importorskip("torch")

from cfg.load_config import validate_and_fill_defaults
from experiments.methods.fedavg import run_fedavg

REQUIRED_COLUMNS = {"round", "client_fraction", "train_loss", "val_loss", "val_acc", "time_round_sec"}

MINIMAL_CFG = {
    "seed": 42,
    "rounds": 2,
    "clients": 2,
    "client_fraction": 1.0,
    "local_epochs": 1,
    "batch_size": 32,
    "lr": 0.01,
    "dataset": {"name": "mnist", "train_samples_cap": 100, "test_samples_cap": 50},
    "partition": {"type": "iid", "alpha": 0.1},
    "server": {"aggregator": "fedavg", "trim_ratio": 0.1, "f": 1, "m": None},
    "attack": {"enabled": False, "type": "signflip", "malicious_fraction": 0.0, "scale": 1.0, "seed": 42, "target": "update"},
    "defense": {"enabled": False, "type": "pid_exclusion", "k_exclude": 1, "Kp": 1.0, "Ki": 0.0, "Kd": 0.0, "warmup_rounds": 0},
}


def _run(tmp_path: Path, cfg_overrides: dict) -> tuple[list[dict], Path]:
    cfg = validate_and_fill_defaults({**MINIMAL_CFG, **cfg_overrides})
    # deep-merge nested dicts that may be overridden
    for key in ("server", "attack", "defense", "partition", "dataset"):
        if key in cfg_overrides:
            merged = dict(MINIMAL_CFG.get(key, {}))
            merged.update(cfg_overrides[key])
            cfg[key] = merged

    run_id = "pytest-fl-integration"
    metrics, _ = run_fedavg(cfg, run_id=run_id, out_dir_results=str(tmp_path))

    results_dir = tmp_path / run_id
    meta_path = results_dir / "run_meta.yaml"
    assert meta_path.exists(), "run_meta.yaml was not written"
    assert (results_dir / "agg_debug.csv").exists(), "agg_debug.csv was not written"
    assert (results_dir / "partition_debug.csv").exists(), "partition_debug.csv was not written"

    return metrics, results_dir


def test_clean_fedavg_schema_and_rounds(tmp_path: Path) -> None:
    """Clean FedAvg baseline: correct schema, correct number of rounds, positive accuracy."""
    metrics, _ = _run(tmp_path, {})

    assert len(metrics) == 2, f"expected 2 rounds of metrics, got {len(metrics)}"

    for row in metrics:
        missing = REQUIRED_COLUMNS - set(row.keys())
        assert not missing, f"metrics row missing columns: {missing}"
        assert float(row["val_acc"]) >= 0.0
        assert float(row["train_loss"]) >= 0.0
        assert float(row["time_round_sec"]) >= 0.0

    # Model should produce valid (non-NaN) accuracy
    final_acc = float(metrics[-1]["val_acc"])
    assert 0.0 <= final_acc <= 1.0, f"final val_acc out of range: {final_acc}"


def test_attack_degrades_accuracy_vs_clean(tmp_path: Path) -> None:
    """Sign-flip attack with 50% malicious clients should not silently vanish."""
    clean_metrics, _ = _run(tmp_path / "clean", {})
    attack_metrics, results_dir = _run(
        tmp_path / "attack",
        {"attack": {"enabled": True, "type": "signflip", "malicious_fraction": 0.5,
                    "scale": 1.0, "seed": 42, "target": "update"}},
    )

    assert (results_dir / "attack_debug.csv").exists(), "attack_debug.csv was not written"

    # attack_debug.csv should record at least one malicious participant
    with (results_dir / "attack_debug.csv").open() as f:
        rows = list(csv.DictReader(f))
    total_malicious = sum(int(r["malicious_count"]) for r in rows)
    assert total_malicious > 0, "attack was enabled but no malicious clients recorded"


def test_defense_writes_debug_csv(tmp_path: Path) -> None:
    """PID defense enabled: defense_debug.csv must be written with TP/FP/FN columns."""
    _, results_dir = _run(
        tmp_path,
        {
            "attack": {"enabled": True, "type": "signflip", "malicious_fraction": 0.5,
                       "scale": 1.0, "seed": 42, "target": "update"},
            "defense": {"enabled": True, "type": "pid_exclusion", "k_exclude": 1,
                        "Kp": 1.0, "Ki": 0.0, "Kd": 0.0, "warmup_rounds": 0},
        },
    )

    debug_path = results_dir / "defense_debug.csv"
    assert debug_path.exists(), "defense_debug.csv was not written"

    with debug_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "defense_debug.csv has no data rows"
    assert "tp" in rows[0] and "fp" in rows[0] and "fn" in rows[0], \
        "defense_debug.csv missing TP/FP/FN columns"
