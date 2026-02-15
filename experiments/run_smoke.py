"""Fast, config-driven smoke run for the FL robustness harness.

This is a lightweight end-to-end experiment stub intended for reproducibility
checks and CI. It does not run full Flower simulation; instead it produces a
small deterministic metrics trace and a simple plot artifact.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

# Add project root to import path when executed as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from cfg.load_config import load as load_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny reproducible smoke experiment")
    parser.add_argument("--config", type=str, default=None, help="Optional config path")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (default: timestamped)")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directories for run-id before writing artifacts",
    )
    return parser.parse_args()


def build_smoke_settings(cfg: dict) -> dict:
    """Derive fast smoke settings from project config values."""
    return {
        "clients": max(2, min(3, int(cfg.get("clients", 3)))),
        "rounds": max(2, min(3, int(cfg.get("rounds", 3)))),
        "seed": int(cfg.get("seed", 42)),
    }


def run_smoke(settings: dict) -> list[dict]:
    """Create deterministic toy metrics for smoke testing."""
    rounds = settings["rounds"]
    clients = settings["clients"]
    rng = np.random.default_rng(settings["seed"])

    metrics = []
    accuracy = 0.45
    loss = 1.25
    for rnd in range(1, rounds + 1):
        accuracy = min(0.99, accuracy + float(rng.uniform(0.03, 0.07)))
        loss = max(0.05, loss - float(rng.uniform(0.08, 0.15)))
        excluded = int(rng.integers(0, max(1, clients // 2)))
        metrics.append(
            {
                "round": rnd,
                "num_clients": clients,
                "accuracy": round(accuracy, 6),
                "loss": round(loss, 6),
                "excluded_clients": excluded,
                "method": "baseline_smoke",
            }
        )
    return metrics


def write_metrics(metrics: list[dict], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def write_plot(metrics: list[dict], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    rounds = [row["round"] for row in metrics]
    accuracy = [row["accuracy"] for row in metrics]

    plt.figure(figsize=(5, 3))
    plt.plot(rounds, accuracy, marker="o")
    plt.title("Smoke Run Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    settings = build_smoke_settings(cfg)

    run_id = args.run_id or f"smoke-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    results_root = Path(cfg["results_dir"]) / run_id
    plots_root = Path(cfg["plots_dir"]) / run_id

    if args.clean:
        shutil.rmtree(results_root, ignore_errors=True)
        shutil.rmtree(plots_root, ignore_errors=True)

    metrics = run_smoke(settings)
    metrics_path = results_root / "metrics.csv"
    plot_path = plots_root / "accuracy.png"

    write_metrics(metrics, metrics_path)
    write_plot(metrics, plot_path)

    print(f"run_id={run_id}")
    print(f"metrics={metrics_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
