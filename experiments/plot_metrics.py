"""Plotting utilities for metrics.csv (single method per plot)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(metrics_csv_path: str | Path) -> list[dict]:
    with Path(metrics_csv_path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_single_method(
    metrics_csv_path: str | Path,
    out_dir: str | Path,
    method: str,
    dpi: int = 120,
    figsize: tuple[float, float] = (6, 4),
) -> Path:
    rows = load_metrics(metrics_csv_path)
    if not rows:
        raise ValueError("metrics.csv is empty")

    rounds = [int(r["round"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method}_val_acc.png"

    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(rounds, val_acc, marker="o")
    plt.title(f"Validation Accuracy ({method})")
    plt.xlabel("Round")
    plt.ylabel("val_acc")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot single-method metrics.csv")
    parser.add_argument("--in", dest="inp", required=True, help="Path to metrics.csv")
    parser.add_argument("--out", required=True, help="Output plot directory")
    parser.add_argument("--method", required=True, help="Method label for output filename/title")
    args = parser.parse_args()

    out_path = plot_single_method(args.inp, args.out, args.method)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
