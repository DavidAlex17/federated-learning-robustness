from __future__ import annotations

import csv
from pathlib import Path

import yaml

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


def test_summarize_runs_and_overwrite(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "run-a"

    _write_metrics(run_dir / "metrics.csv", val_acc=0.75)
    with (run_dir / "run_meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "method": "fedavg_tiny",
                "aggregator": "fedavg",
                "dataset_used": "mnist",
                "seed": 42,
                "rounds": 1,
                "clients": 4,
                "client_fraction": 1.0,
                "attack": {"enabled": True, "type": "signflip", "malicious_fraction": 0.5, "malicious_ids_count": 2},
                "defense": {"enabled": True, "type": "pid_exclusion", "k_exclude": 1, "Kp": 1.0, "Ki": 0.0, "Kd": 0.0, "warmup_rounds": 0},
            },
            f,
            sort_keys=False,
        )

    with (run_dir / "defense_debug.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "n_total", "n_excluded", "excluded_ids", "malicious_ids_in_round", "tp", "fp", "fn", "precision", "recall", "scores"],
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
                "precision": 1.0,
                "recall": 0.5,
                "scores": "",
            }
        )

    out_path = results_root / "summary.csv"
    rows = summarize_runs(results_root=results_root, out_path=out_path)
    assert len(rows) == 1
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert written[0]["run_id"] == "run-a"
    assert written[0]["final_val_acc"] == "0.75"
    assert written[0]["defense_avg_precision"] == "1.000000"

    # overwrite behavior: change metrics and regenerate
    _write_metrics(run_dir / "metrics.csv", val_acc=0.85)
    summarize_runs(results_root=results_root, out_path=out_path)
    with out_path.open("r", encoding="utf-8") as f:
        rewritten = list(csv.DictReader(f))
    assert rewritten[0]["final_val_acc"] == "0.85"
