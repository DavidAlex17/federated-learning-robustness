"""FL run entry point: wires data, model, client, strategy, and debug writers."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import yaml

from data.mnist import _build_partitioned_data
from data.partition import _dirichlet_partition_indices, _partition_indices_iid
from framework.attack import apply_signflip_attack, select_malicious_client_ids
from framework.client import MnistClient
from framework.defense import cosine_direction_error, select_top_k_by_score, update_pid_score
from framework.strategy import RobustFedStrategy


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_agg_debug(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["round", "aggregator", "n_clients_total", "n_clients_used", "n_selected", "trim_ratio"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_attack_debug(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["round", "malicious_fraction", "malicious_count", "total_clients_sampled", "malicious_ids_participated"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_partition_debug(path: Path, train_parts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["client_id", "n_samples"] + [f"label_{i}_count" for i in range(10)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cid, subset in enumerate(train_parts):
            counts = [0] * 10
            for _, label in subset:
                if 0 <= int(label) <= 9:
                    counts[int(label)] += 1
            row = {"client_id": cid, "n_samples": len(subset)}
            for i in range(10):
                row[f"label_{i}_count"] = counts[i]
            writer.writerow(row)


def _write_defense_debug(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "round", "n_total", "n_excluded", "excluded_ids", "malicious_ids_in_round",
        "tp", "fp", "fn", "precision", "recall", "scores",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_meta(
    path: Path,
    config: dict,
    dataset_used: str,
    aggregator_cfg: dict,
    attack_cfg: dict,
    defense_cfg: dict,
    malicious_ids: set[int],
    partition_cfg: dict,
) -> None:
    dataset_cfg = config.get("dataset", {})
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "fedavg",
        "aggregator": aggregator_cfg.get("aggregator", "fedavg"),
        "trim_ratio": aggregator_cfg.get("trim_ratio", 0.1),
        "f": aggregator_cfg.get("f", 1),
        "m": aggregator_cfg.get("m", None),
        "dataset_used": dataset_used,
        "seed": int(config.get("seed", 42)),
        "rounds": int(config.get("rounds", 3)),
        "clients": int(config.get("clients", 4)),
        "client_fraction": float(config.get("client_fraction", 1.0)),
        "train_samples_cap": int(dataset_cfg.get("train_samples_cap", 1000)),
        "test_samples_cap": int(dataset_cfg.get("test_samples_cap", 200)),
        "partition": {"type": str(partition_cfg.get("type", "iid")), "alpha": float(partition_cfg.get("alpha", 0.1))},
        "attack": {
            "enabled": bool(attack_cfg.get("enabled", False)),
            "type": str(attack_cfg.get("type", "signflip")),
            "target": str(attack_cfg.get("target", "update")),
            "malicious_fraction": float(attack_cfg.get("malicious_fraction", 0.0)),
            "scale": float(attack_cfg.get("scale", 1.0)),
            "seed": int(attack_cfg.get("seed", config.get("seed", 42))),
            "malicious_ids_count": len(malicious_ids),
            "malicious_ids": sorted(malicious_ids),
        },
        "defense": {
            "enabled": bool(defense_cfg.get("enabled", False)),
            "type": str(defense_cfg.get("type", "pid_exclusion")),
            "k_exclude": int(defense_cfg.get("k_exclude", 1)),
            "Kp": float(defense_cfg.get("Kp", 1.0)),
            "Ki": float(defense_cfg.get("Ki", 0.0)),
            "Kd": float(defense_cfg.get("Kd", 0.0)),
            "warmup_rounds": int(defense_cfg.get("warmup_rounds", 0)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def run_fedavg(config: dict, run_id: str, out_dir_results: str) -> tuple[list[dict], str]:
    """Run FL with configurable robust aggregation; return metrics and plot tag."""
    _set_global_seed(int(config.get("seed", 42)))

    rounds = int(config.get("rounds", 2))
    num_clients = int(config.get("clients", 4))
    client_fraction = float(config.get("client_fraction", 1.0))
    fit_clients = max(1, int(np.ceil(num_clients * client_fraction)))
    server_cfg = config.get("server", {})
    aggregator = server_cfg.get("aggregator", "fedavg")
    partition_cfg = config.get("partition", {})

    attack_cfg = config.get("attack", {})
    attack_seed = int(attack_cfg.get("seed", config.get("seed", 42)))
    malicious_ids = select_malicious_client_ids(
        num_clients=num_clients,
        malicious_fraction=float(attack_cfg.get("malicious_fraction", 0.0)),
        seed=attack_seed,
    )
    attack_enabled = bool(attack_cfg.get("enabled", False))
    print(
        f"attack enabled={attack_enabled} type={attack_cfg.get('type', 'signflip')} "
        f"fraction={float(attack_cfg.get('malicious_fraction', 0.0))} malicious_ids={sorted(malicious_ids)}"
    )

    defense_cfg = config.get("defense", {})
    print(
        f"defense enabled={bool(defense_cfg.get('enabled', False))} "
        f"type={defense_cfg.get('type', 'pid_exclusion')} k_exclude={int(defense_cfg.get('k_exclude', 1))}"
    )

    data = _build_partitioned_data(config)

    def client_fn(context_or_cid):
        if hasattr(context_or_cid, "node_config"):
            cid = int(context_or_cid.node_config.get("partition-id", 0))
        else:
            cid = int(context_or_cid)
        client = MnistClient(
            cid=cid,
            train_subset=data.train_parts[cid],
            test_subset=data.test_set,
            config=config,
            malicious_ids=malicious_ids,
            attack_cfg=attack_cfg,
        )
        return client.to_client() if hasattr(client, "to_client") else client

    strategy = RobustFedStrategy(
        client_fraction=client_fraction,
        aggregator_cfg=server_cfg,
        attack_cfg=attack_cfg,
        defense_cfg=defense_cfg,
        malicious_ids=malicious_ids,
        fraction_fit=client_fraction,
        fraction_evaluate=client_fraction,
        min_fit_clients=fit_clients,
        min_evaluate_clients=fit_clients,
        min_available_clients=num_clients,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    results_root = Path(out_dir_results) / run_id
    _write_run_meta(
        results_root / "run_meta.yaml",
        config=config,
        dataset_used=data.dataset_used,
        aggregator_cfg=server_cfg,
        attack_cfg=attack_cfg,
        defense_cfg=defense_cfg,
        malicious_ids=malicious_ids,
        partition_cfg=partition_cfg,
    )
    _write_partition_debug(results_root / "partition_debug.csv", data.train_parts)
    _write_agg_debug(results_root / "agg_debug.csv", strategy.agg_debug)
    _write_attack_debug(results_root / "attack_debug.csv", strategy.attack_debug)
    _write_defense_debug(results_root / "defense_debug.csv", strategy.defense_debug)

    return strategy.round_metrics, f"fedavg_{aggregator}"

