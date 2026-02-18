"""Fast, config-driven smoke run for the FL robustness harness."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import sys

# Add project root to import path when executed as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from cfg.load_config import load as load_cfg
from experiments.runner import run_experiment


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
    cfg = dict(cfg)
    cfg["clients"] = max(2, min(3, int(cfg.get("clients", 3))))
    cfg["rounds"] = max(2, min(3, int(cfg.get("rounds", 3))))
    cfg["client_fraction"] = min(1.0, max(0.1, float(cfg.get("client_fraction", 1.0))))
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_smoke_settings(load_cfg(args.config))

    run_id = args.run_id or f"smoke-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    results_root = Path(cfg["results_dir"]) / run_id
    plots_root = Path(cfg["plots_dir"]) / run_id

    if args.clean:
        shutil.rmtree(results_root, ignore_errors=True)
        shutil.rmtree(plots_root, ignore_errors=True)

    run_experiment(
        config=cfg,
        run_id=run_id,
        out_dir_results=cfg["results_dir"],
        out_dir_plots=cfg["plots_dir"],
        method="smoke_synth",
    )

    print(f"run_id={run_id}")
    print(f"metrics={results_root / 'metrics.csv'}")
    print(f"plot={plots_root / 'smoke_synth_val_acc.png'}")


if __name__ == "__main__":
    main()
