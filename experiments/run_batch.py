"""Batch experiment runner.

Reads a YAML file of run definitions, merges each entry's overrides on top of
the base config from cfg/load_config.py, and calls run_experiment() for each.

Usage:
    PYTHONPATH=. python experiments/run_batch.py
    PYTHONPATH=. python experiments/run_batch.py --batch experiments/batch_runs.yaml
    PYTHONPATH=. python experiments/run_batch.py --batch my_custom_batch.yaml
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import yaml

from cfg.load_config import load
from experiments.plot_aggregator_comparison import plot_aggregator_comparison
from experiments.plot_comparison import plot_comparison
from experiments.runner import run_experiment

_DEFAULT_BATCH = Path(__file__).parent / "batch_runs.yaml"


def _apply_overrides(cfg: dict, run_def: dict) -> dict:
    """Return a deep copy of cfg with run_def overrides applied."""
    cfg = copy.deepcopy(cfg)

    if "rounds" in run_def:
        cfg["rounds"] = int(run_def["rounds"])

    if "clients" in run_def:
        cfg["clients"] = int(run_def["clients"])

    if "partition" in run_def:
        part = run_def["partition"]
        cfg.setdefault("partition", {})
        if "type" in part:
            cfg["partition"]["type"] = part["type"]
        if "alpha" in part:
            cfg["partition"]["alpha"] = float(part["alpha"])

    if "attack" in run_def:
        atk = run_def["attack"]
        cfg.setdefault("attack", {})
        if "enabled" in atk:
            cfg["attack"]["enabled"] = bool(atk["enabled"])
        if "type" in atk:
            cfg["attack"]["type"] = atk["type"]
        if "malicious_fraction" in atk:
            cfg["attack"]["malicious_fraction"] = float(atk["malicious_fraction"])

    if "defense" in run_def:
        dfn = run_def["defense"]
        cfg.setdefault("defense", {})
        if "enabled" in dfn:
            cfg["defense"]["enabled"] = bool(dfn["enabled"])
        if "type" in dfn:
            cfg["defense"]["type"] = dfn["type"]
        if "k_exclude" in dfn:
            cfg["defense"]["k_exclude"] = int(dfn["k_exclude"])
        if "Kp" in dfn:
            cfg["defense"]["Kp"] = float(dfn["Kp"])

    if "aggregator" in run_def:
        cfg.setdefault("server", {})["aggregator"] = str(run_def["aggregator"])

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of FL experiments.")
    parser.add_argument(
        "--batch",
        default=str(_DEFAULT_BATCH),
        help="Path to a YAML file defining the batch of runs (default: experiments/batch_runs.yaml).",
    )
    args = parser.parse_args()

    batch_path = Path(args.batch)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    with batch_path.open(encoding="utf-8") as fh:
        batch = yaml.safe_load(fh)

    runs = batch.get("runs", [])
    if not runs:
        print("No runs defined in batch file. Exiting.")
        return

    base_cfg = load()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir_results = str(Path(base_cfg["results_dir"]) / timestamp)
    out_dir_plots = str(Path(base_cfg["plots_dir"]) / timestamp)
    Path(out_dir_results).mkdir(parents=True, exist_ok=True)
    Path(out_dir_plots).mkdir(parents=True, exist_ok=True)
    print(f"Batch folder: {timestamp}")

    total = len(runs)
    for idx, run_def in enumerate(runs, start=1):
        run_id = run_def.get("run_id")
        if not run_id:
            raise ValueError(f"Run entry {idx} is missing a 'run_id' field.")

        print(f"Run {idx}/{total}: {run_id}")
        cfg = _apply_overrides(base_cfg, run_def)
        run_experiment(
            config=cfg,
            run_id=run_id,
            out_dir_results=out_dir_results,
            out_dir_plots=out_dir_plots,
            method="fedavg",
        )

    print(f"Batch complete. {total} runs finished.")

    print("Generating comparison plots...")
    plot_comparison(
        results_root=Path(out_dir_results),
        out_dir=Path(out_dir_plots),
    )
    print(f"Comparison plots saved to {out_dir_plots}")

    print("Generating aggregator comparison plots...")
    plot_aggregator_comparison(
        results_root=Path(out_dir_results),
        out_dir=Path(out_dir_plots),
    )
    print(f"Aggregator comparison plots saved to {out_dir_plots}")


if __name__ == "__main__":
    main()
