"""Reusable lightweight experiment runner for synthetic and FL experiments."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from experiments.methods import run_fedavg_tiny
from experiments.plot_metrics import plot_single_method

METRIC_COLUMNS = [
    "round",
    "client_fraction",
    "train_loss",
    "val_loss",
    "val_acc",
    "time_round_sec",
]


def _write_metrics_csv(metrics: list[dict], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(metrics)


def _run_smoke_synth(config: dict) -> list[dict]:
    rounds = int(config.get("rounds", 3))
    seed = int(config.get("seed", 42))
    client_fraction = float(config.get("client_fraction", 1.0))
    rng = np.random.default_rng(seed)

    metrics = []
    val_acc = 0.45
    train_loss = 1.20
    val_loss = 1.35

    for rnd in range(1, rounds + 1):
        t0 = time.perf_counter()
        val_acc = min(0.99, val_acc + float(rng.uniform(0.03, 0.07)))
        train_loss = max(0.05, train_loss - float(rng.uniform(0.08, 0.14)))
        val_loss = max(0.05, val_loss - float(rng.uniform(0.07, 0.12)))
        elapsed = time.perf_counter() - t0

        metrics.append(
            {
                "round": rnd,
                "client_fraction": round(client_fraction, 6),
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "time_round_sec": round(elapsed, 6),
            }
        )
    return metrics


def run_experiment(
    config: dict,
    run_id: str,
    out_dir_results: str,
    out_dir_plots: str,
    method: str,
) -> None:
    """Run selected method and emit metrics + one method plot."""
    plot_method = method
    if method == "smoke_synth":
        metrics = _run_smoke_synth(config)
    elif method == "fedavg_tiny":
        metrics, plot_method = run_fedavg_tiny(config, run_id=run_id, out_dir_results=out_dir_results)
    else:
        raise ValueError(f"Unsupported method: {method}")

    results_root = Path(out_dir_results) / run_id
    plots_root = Path(out_dir_plots) / run_id
    metrics_path = results_root / "metrics.csv"

    _write_metrics_csv(metrics, metrics_path)
    plot_single_method(
        metrics_csv_path=metrics_path,
        out_dir=plots_root,
        method=plot_method,
        dpi=int(config.get("plot", {}).get("dpi", 120)),
        figsize=tuple(config.get("plot", {}).get("figsize", [6, 4])),
    )
