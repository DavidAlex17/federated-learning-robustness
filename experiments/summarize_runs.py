"""Summarize experiment run artifacts into a single CSV report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import yaml

SUMMARY_COLUMNS = [
    "run_id",
    "method",
    "aggregator",
    "dataset_used",
    "seed",
    "rounds",
    "clients",
    "client_fraction",
    "partition_type",
    "partition_alpha",
    "attack_enabled",
    "attack_type",
    "malicious_fraction",
    "malicious_ids_count",
    "defense_enabled",
    "defense_type",
    "k_exclude",
    "Kp",
    "Ki",
    "Kd",
    "warmup_rounds",
    "final_val_acc",
    "final_val_loss",
    "final_train_loss",
    "avg_time_round_sec",
    "defense_avg_precision",
    "defense_avg_recall",
    "defense_avg_tp",
    "defense_avg_fp",
    "defense_avg_fn",
]

MINIMAL_COLUMNS = [
    "run_id",
    "aggregator",
    "partition_type",
    "partition_alpha",
    "malicious_fraction",
    "defense_enabled",
    "final_val_acc",
    "defense_avg_precision",
    "defense_avg_recall",
]


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    return f"{sum(vals) / len(vals):.6f}"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_metrics(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    required = {"round", "val_acc", "val_loss", "train_loss", "time_round_sec"}
    if not rows or not required.issubset(set(rows[0].keys())):
        return None
    return rows


def _final_metrics(rows: list[dict]) -> tuple[str, str, str, str]:
    def round_key(row):
        return _as_float(row.get("round")) if _as_float(row.get("round")) is not None else -1

    final = max(rows, key=round_key)
    avg_time = _mean(_as_float(r.get("time_round_sec")) for r in rows)
    return (
        final.get("val_acc", ""),
        final.get("val_loss", ""),
        final.get("train_loss", ""),
        avg_time,
    )


def _defense_summary(path: Path) -> tuple[str, str, str, str, str]:
    if not path.exists():
        return "", "", "", "", ""
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "", "", "", "", ""
    return (
        _mean(_as_float(r.get("precision")) for r in rows),
        _mean(_as_float(r.get("recall")) for r in rows),
        _mean(_as_float(r.get("tp")) for r in rows),
        _mean(_as_float(r.get("fp")) for r in rows),
        _mean(_as_float(r.get("fn")) for r in rows),
    )


def summarize_runs(
    results_root: Path,
    out_path: Path,
    runs: list[str] | None = None,
    glob_pattern: str | None = None,
    write_output: bool = True,
) -> list[dict]:
    run_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    if runs:
        allowed = set(runs)
        run_dirs = [p for p in run_dirs if p.name in allowed]
    if glob_pattern:
        run_dirs = [p for p in run_dirs if p.match(glob_pattern)]

    summary_rows: list[dict] = []
    for run_dir in sorted(run_dirs, key=lambda p: p.name):
        metrics = _load_metrics(run_dir / "metrics.csv")
        if metrics is None:
            print(f"[summarize] warning: skipping malformed or missing metrics for {run_dir.name}")
            continue

        meta = _load_yaml(run_dir / "run_meta.yaml")
        attack = meta.get("attack", {}) if isinstance(meta.get("attack", {}), dict) else {}
        defense = meta.get("defense", {}) if isinstance(meta.get("defense", {}), dict) else {}
        partition = meta.get("partition", {}) if isinstance(meta.get("partition", {}), dict) else {}

        final_val_acc, final_val_loss, final_train_loss, avg_time = _final_metrics(metrics)
        d_prec, d_rec, d_tp, d_fp, d_fn = _defense_summary(run_dir / "defense_debug.csv")

        row = {
            "run_id": run_dir.name,
            "method": meta.get("method", ""),
            "aggregator": meta.get("aggregator", ""),
            "dataset_used": meta.get("dataset_used", ""),
            "seed": meta.get("seed", ""),
            "rounds": meta.get("rounds", ""),
            "clients": meta.get("clients", ""),
            "client_fraction": meta.get("client_fraction", ""),
            "partition_type": partition.get("type", ""),
            "partition_alpha": partition.get("alpha", ""),
            "attack_enabled": attack.get("enabled", ""),
            "attack_type": attack.get("type", ""),
            "malicious_fraction": attack.get("malicious_fraction", ""),
            "malicious_ids_count": attack.get("malicious_ids_count", ""),
            "defense_enabled": defense.get("enabled", ""),
            "defense_type": defense.get("type", ""),
            "k_exclude": defense.get("k_exclude", ""),
            "Kp": defense.get("Kp", ""),
            "Ki": defense.get("Ki", ""),
            "Kd": defense.get("Kd", ""),
            "warmup_rounds": defense.get("warmup_rounds", ""),
            "final_val_acc": final_val_acc,
            "final_val_loss": final_val_loss,
            "final_train_loss": final_train_loss,
            "avg_time_round_sec": avg_time,
            "defense_avg_precision": d_prec,
            "defense_avg_recall": d_rec,
            "defense_avg_tp": d_tp,
            "defense_avg_fp": d_fp,
            "defense_avg_fn": d_fn,
        }
        summary_rows.append(row)

    if write_output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)

    return summary_rows


def _format_minimal_table(rows: list[dict], limit: int) -> str:
    shown = rows[: max(0, limit)]
    data = [MINIMAL_COLUMNS]
    for row in shown:
        data.append([str(row.get(col, "")) for col in MINIMAL_COLUMNS])

    widths = [max(len(str(r[i])) for r in data) for i in range(len(MINIMAL_COLUMNS))]

    lines = []
    header = " | ".join(str(MINIMAL_COLUMNS[i]).ljust(widths[i]) for i in range(len(MINIMAL_COLUMNS)))
    sep = "-+-".join("-" * widths[i] for i in range(len(MINIMAL_COLUMNS)))
    lines.append(header)
    lines.append(sep)
    for row in shown:
        line = " | ".join(str(row.get(MINIMAL_COLUMNS[i], "")).ljust(widths[i]) for i in range(len(MINIMAL_COLUMNS)))
        lines.append(line)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize run artifacts into a single CSV")
    parser.add_argument("--results-root", default="experiments/results", help="Results root directory")
    parser.add_argument("--out", default="experiments/results/summary.csv", help="Output summary CSV")
    parser.add_argument("--runs", nargs="*", default=None, help="Optional explicit run-id filter list")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help="Optional run-id glob filter")
    parser.add_argument("--minimal", action="store_true", help="Print a compact fixed-width table")
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows to display in minimal mode")
    parser.add_argument("--no-write", action="store_true", help="Do not write summary CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_runs(
        results_root=Path(args.results_root),
        out_path=Path(args.out),
        runs=args.runs,
        glob_pattern=args.glob_pattern,
        write_output=not args.no_write,
    )
    if args.minimal:
        print(_format_minimal_table(rows, limit=args.limit))
    print(f"summarized_runs={len(rows)}")
    print(f"summary_path={Path(args.out)}")


if __name__ == "__main__":
    main()
