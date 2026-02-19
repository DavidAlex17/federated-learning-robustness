"""Run tiny FedAvg backend (Flower + PyTorch) with run-scoped outputs."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cfg.load_config import load as load_cfg
from experiments.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny FedAvg experiment")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier")
    parser.add_argument("--clean", action="store_true", help="Delete run outputs before writing")
    parser.add_argument("--rounds", type=int, default=None, help="Optional rounds override")
    parser.add_argument("--aggregator", type=str, default=None, help="Optional aggregator override")
    parser.add_argument("--attack", action="store_true", help="Enable sign-flip attack")
    parser.add_argument("--malicious-fraction", type=float, default=None, help="Override attack malicious fraction")
    parser.add_argument("--attack-scale", type=float, default=None, help="Override attack scale")
    parser.add_argument("--config", type=str, default=None, help="Optional config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.aggregator is not None:
        cfg.setdefault("server", {})["aggregator"] = args.aggregator
    if args.attack:
        cfg.setdefault("attack", {})["enabled"] = True
    if args.malicious_fraction is not None:
        cfg.setdefault("attack", {})["malicious_fraction"] = args.malicious_fraction
    if args.attack_scale is not None:
        cfg.setdefault("attack", {})["scale"] = args.attack_scale

    run_id = args.run_id or f"fedavg-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
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
        method="fedavg_tiny",
    )

    print(f"run_id={run_id}")
    print(f"metrics={results_root / 'metrics.csv'}")
    agg = cfg.get('server', {}).get('aggregator', 'fedavg')
    print(f"plot={plots_root / f'fedavg_tiny_{agg}_val_acc.png'}")


if __name__ == "__main__":
    main()
