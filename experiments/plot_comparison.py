"""
Generate a side-by-side comparison plot for the 6 canonical IID/non-IID runs.

Produces two figures:
  experiments/plots/comparison_val_acc.png   — validation accuracy
  experiments/plots/comparison_val_loss.png  — validation loss

Each figure has two panels: IID (left) and non-IID (right), with three lines
per panel: clean baseline, attack only, attack + PID defense.

Usage:
    PYTHONPATH=. python experiments/plot_comparison.py
    PYTHONPATH=. python experiments/plot_comparison.py --results-root experiments/results --out experiments/plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = {
    "iid": {
        "clean":   "iid-clean",
        "attack":  "iid-atk25",
        "defense": "iid-pid-atk25",
    },
    "niid": {
        "clean":   "niid-clean",
        "attack":  "niid-atk25",
        "defense": "niid-pid-atk25",
    },
}

LINE_STYLE = {
    "clean":   {"color": "steelblue",  "linestyle": "-",  "marker": "o", "label": "Clean (no attack)"},
    "attack":  {"color": "firebrick",  "linestyle": "--", "marker": "s", "label": "Attack only (25% sign-flip)"},
    "defense": {"color": "seagreen",   "linestyle": "-.", "marker": "^", "label": "Attack + PID defense"},
}


def _load(results_root: Path, run_id: str) -> list[dict]:
    csv_path = results_root / run_id / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col(rows: list[dict], col: str) -> tuple[list[int], list[float]]:
    rounds = [int(r["round"]) for r in rows]
    values = [float(r[col]) for r in rows]
    return rounds, values


def plot_comparison(results_root: Path, out_dir: Path, dpi: int = 130) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, title_metric in [
        ("val_acc",  "Validation Accuracy", "Validation Accuracy"),
        ("val_loss", "Validation Loss",     "Validation Loss"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
        fig.suptitle(
            f"{title_metric} — IID vs non-IID  |  FedAvg, 25% sign-flip attack",
            fontsize=10, y=1.01,
        )

        for ax, (partition_key, runs) in zip(axes, RUNS.items()):
            partition_label = "IID" if partition_key == "iid" else "non-IID (Dirichlet α=0.1)"
            ax.set_title(partition_label, fontsize=10)

            for condition, run_id in runs.items():
                rows = _load(results_root, run_id)
                x, y = _col(rows, metric)
                style = LINE_STYLE[condition]
                ax.plot(x, y, **style, linewidth=1.8, markersize=6)

            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 2, 3])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = out_dir / f"comparison_{metric}.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IID vs non-IID comparison figures")
    parser.add_argument("--results-root", default="experiments/results", help="Root of run result directories")
    parser.add_argument("--out", default="experiments/plots", help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()

    plot_comparison(
        results_root=Path(args.results_root),
        out_dir=Path(args.out),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
