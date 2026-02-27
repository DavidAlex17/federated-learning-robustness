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
    parser.add_argument("--defense", action="store_true", help="Enable PID-style exclusion defense")
    parser.add_argument("--k-exclude", type=int, default=None, help="Defense top-k exclusions")
    parser.add_argument("--kp", type=float, default=None, help="Defense Kp")
    parser.add_argument("--ki", type=float, default=None, help="Defense Ki")
    parser.add_argument("--kd", type=float, default=None, help="Defense Kd")
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default=None, help="Optional partition type override")
    parser.add_argument("--alpha", type=float, default=None, help="Optional Dirichlet alpha override")
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

    if args.defense:
        cfg.setdefault("defense", {})["enabled"] = True
    if args.k_exclude is not None:
        cfg.setdefault("defense", {})["k_exclude"] = args.k_exclude
    if args.kp is not None:
        cfg.setdefault("defense", {})["Kp"] = args.kp
    if args.ki is not None:
        cfg.setdefault("defense", {})["Ki"] = args.ki
    if args.kd is not None:
        cfg.setdefault("defense", {})["Kd"] = args.kd

    if args.partition is not None:
        cfg.setdefault("partition", {})["type"] = args.partition
    if args.alpha is not None:
        cfg.setdefault("partition", {})["alpha"] = args.alpha

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
