from __future__ import annotations

import csv
from pathlib import Path

import yaml

from experiments import summarize_runs as summarize_module
from experiments.summarize_runs import summarize_runs


def _write_metrics(path: Path, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "client_fraction", "train_loss", "val_loss", "val_acc", "time_round_sec"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "round": 1,
                "client_fraction": 1.0,
                "train_loss": 1.2,
                "val_loss": 1.0,
                "val_acc": val_acc,
                "time_round_sec": 0.5,
            }
        )


def _write_meta(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "method": "fedavg",
                "aggregator": "fedavg",
                "dataset_used": "mnist",
                "seed": 42,
                "rounds": 1,
                "clients": 4,
                "client_fraction": 1.0,
                "partition": {"type": "dirichlet", "alpha": 0.1},
                "attack": {"enabled": True, "type": "signflip", "malicious_fraction": 0.5, "malicious_ids_count": 2},
                "defense": {
                    "enabled": True,
                    "type": "pid_exclusion",
                    "k_exclude": 1,
                    "Kp": 1.0,
                    "Ki": 0.0,
                    "Kd": 0.0,
                    "warmup_rounds": 0,
                },
            },
            f,
            sort_keys=False,
        )


def _write_defense(path: Path, precision: str = "1.0", recall: str = "0.5") -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "n_total",
                "n_excluded",
                "excluded_ids",
                "malicious_ids_in_round",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "scores",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "round": 1,
                "n_total": 4,
                "n_excluded": 1,
                "excluded_ids": "2",
                "malicious_ids_in_round": "2|3",
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "precision": precision,
                "recall": recall,
                "scores": "",
            }
        )


def test_summarize_runs_and_overwrite(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "run-a"

    _write_metrics(run_dir / "metrics.csv", val_acc=0.75)
    _write_meta(run_dir / "run_meta.yaml")
    _write_defense(run_dir / "defense_debug.csv")

    out_path = results_root / "summary.csv"
    rows = summarize_runs(results_root=results_root, out_path=out_path)
    assert len(rows) == 1
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert written[0]["run_id"] == "run-a"
    assert written[0]["final_val_acc"] == "0.75"
    assert written[0]["defense_avg_precision"] == "1.000000"
    assert written[0]["partition_type"] == "dirichlet"

    # overwrite behavior: change metrics and regenerate
    _write_metrics(run_dir / "metrics.csv", val_acc=0.85)
    summarize_runs(results_root=results_root, out_path=out_path)
    with out_path.open("r", encoding="utf-8") as f:
        rewritten = list(csv.DictReader(f))
    assert rewritten[0]["final_val_acc"] == "0.85"


def test_minimal_mode_prints_table(tmp_path: Path, monkeypatch, capsys) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "atk-demo"
    _write_metrics(run_dir / "metrics.csv", val_acc=0.66)
    _write_meta(run_dir / "run_meta.yaml")
    _write_defense(run_dir / "defense_debug.csv")

    out_path = results_root / "summary.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_runs.py",
            "--results-root",
            str(results_root),
            "--out",
            str(out_path),
            "--minimal",
            "--limit",
            "10",
            "--glob",
            "atk*",
        ],
    )
    summarize_module.main()
    captured = capsys.readouterr().out

    assert "run_id" in captured
    assert "atk%" in captured
    assert "partition" in captured
    assert "defense" in captured
    assert "precision" in captured
    assert "atk-demo" in captured
    assert out_path.exists()


def test_defense_zero_metrics_are_not_treated_as_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "zero-defense"
    _write_metrics(run_dir / "metrics.csv", val_acc=0.44)
    _write_meta(run_dir / "run_meta.yaml")
    _write_defense(run_dir / "defense_debug.csv", precision="0.0", recall="0")

    out_path = results_root / "summary.csv"
    rows = summarize_runs(results_root=results_root, out_path=out_path)
    assert rows[0]["defense_avg_precision"] == "0.000000"
    assert rows[0]["defense_avg_recall"] == "0.000000"

    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_runs.py",
            "--results-root",
            str(results_root),
            "--out",
            str(out_path),
            "--minimal",
            "--glob",
            "zero*",
        ],
    )
    summarize_module.main()
    captured = capsys.readouterr().out
    # defense is enabled with 0.0 precision — should render as 0.000, not —
    assert "0.000" in captured
    assert "—" not in captured
