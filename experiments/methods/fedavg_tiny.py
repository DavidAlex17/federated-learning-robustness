"""Tiny FL backend using Flower + PyTorch on capped MNIST."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from framework.aggregators import fedavg, multi_krum, trimmed_mean


@dataclass
class PartitionBundle:
    train_parts: list[Subset]
    test_set: Subset
    dataset_used: str


def select_malicious_client_ids(num_clients: int, malicious_fraction: float, seed: int) -> set[int]:
    """Select a deterministic malicious client-id set."""
    malicious_count = int(np.floor(max(0.0, min(1.0, malicious_fraction)) * num_clients))
    rng = np.random.default_rng(seed)
    ids = rng.permutation(num_clients)[:malicious_count]
    return set(int(i) for i in ids)


def apply_signflip_attack(old_params: list[np.ndarray], new_params: list[np.ndarray], scale: float) -> list[np.ndarray]:
    """Apply sign-flip to parameter update: old + (-scale * (new-old))."""
    attacked = []
    for old, new in zip(old_params, new_params):
        update = new - old
        attacked_update = -float(scale) * update
        attacked.append(old + attacked_update)
    return attacked


def update_pid_score(error: float, state: dict, kp: float, ki: float, kd: float) -> tuple[float, dict]:
    """Update PID state from a scalar error and return score and new state."""
    integral = float(state.get("integral", 0.0)) + float(error)
    prev_error = float(state.get("prev_error", 0.0))
    derivative = float(error) - prev_error
    score = kp * float(error) + ki * integral + kd * derivative
    return float(score), {"integral": integral, "prev_error": float(error)}


def select_top_k_by_score(scores: dict[int, float], k: int) -> list[int]:
    """Select top-k ids by descending score, deterministic tie-break by client id."""
    if k <= 0:
        return []
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in ordered[: min(k, len(ordered))]]


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cap_dataset(ds, cap: int, seed: int) -> Subset:
    indices = np.arange(len(ds))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    indices = indices[: min(cap, len(indices))]
    return Subset(ds, indices.tolist())


def _build_partitioned_data(config: dict) -> PartitionBundle:
    dataset_cfg = config.get("dataset", {})
    train_cap = int(dataset_cfg.get("train_samples_cap", 1000))
    test_cap = int(dataset_cfg.get("test_samples_cap", 200))
    num_clients = int(config.get("clients", 4))
    seed = int(config.get("seed", 42))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset_used = "mnist"
    try:
        train_raw = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_raw = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    except RuntimeError:
        dataset_used = "fakedata"
        train_raw = datasets.FakeData(
            size=max(train_cap, 1), image_size=(1, 28, 28), num_classes=10, transform=transform, random_offset=seed
        )
        test_raw = datasets.FakeData(
            size=max(test_cap, 1), image_size=(1, 28, 28), num_classes=10, transform=transform, random_offset=seed + 1
        )

    print(f"dataset={dataset_used}")

    train_subset = _cap_dataset(train_raw, train_cap, seed=seed)
    test_subset = _cap_dataset(test_raw, test_cap, seed=seed + 1)

    rng = np.random.default_rng(seed)
    train_indices = np.array(train_subset.indices)
    rng.shuffle(train_indices)
    split_indices = np.array_split(train_indices, num_clients)
    train_parts = [Subset(train_raw, idxs.tolist()) for idxs in split_indices]

    return PartitionBundle(train_parts=train_parts, test_set=test_subset, dataset_used=dataset_used)


def _get_params(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def _set_params(model: nn.Module, params: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state, strict=True)


def _train_one_epoch(model: nn.Module, loader: DataLoader, lr: float, device: str) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        batch_n = x.size(0)
        loss_sum += loss.item() * batch_n
        n += batch_n
    return loss_sum / max(n, 1)


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            batch_n = x.size(0)
            loss_sum += loss.item() * batch_n
            correct += (preds == y).sum().item()
            n += batch_n
    return loss_sum / max(n, 1), correct / max(n, 1)


class MnistClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_subset: Subset,
        test_subset: Subset,
        config: dict,
        malicious_ids: set[int],
        attack_cfg: dict,
    ) -> None:
        self.cid = cid
        self.is_malicious = cid in malicious_ids and bool(attack_cfg.get("enabled", False))
        self.attack_type = str(attack_cfg.get("type", "signflip"))
        self.attack_scale = float(attack_cfg.get("scale", 1.0))

        self.device = "cpu"
        self.model = TinyMLP().to(self.device)
        self.lr = float(config.get("lr", 0.01))
        self.local_epochs = int(config.get("local_epochs", 1))
        self.batch_size = int(config.get("batch_size", 32))
        self.train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

    def get_parameters(self, config):
        return _get_params(self.model)

    def fit(self, parameters, config):
        old_params = [np.array(p, copy=True) for p in parameters]
        _set_params(self.model, old_params)
        losses = [_train_one_epoch(self.model, self.train_loader, self.lr, self.device) for _ in range(self.local_epochs)]
        mean_loss = float(np.mean(losses)) if losses else 0.0
        new_params = _get_params(self.model)

        if self.is_malicious and self.attack_type == "signflip":
            returned_params = apply_signflip_attack(old_params, new_params, self.attack_scale)
        else:
            returned_params = new_params

        return returned_params, len(self.train_loader.dataset), {
            "train_loss": mean_loss,
            "client_id": self.cid,
            "is_malicious": int(self.is_malicious),
        }

    def evaluate(self, parameters, config):
        _set_params(self.model, parameters)
        val_loss, val_acc = _evaluate(self.model, self.test_loader, self.device)
        return float(val_loss), len(self.test_loader.dataset), {"val_acc": float(val_acc)}


class RobustFedStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        client_fraction: float,
        aggregator_cfg: dict,
        attack_cfg: dict,
        defense_cfg: dict,
        malicious_ids: set[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client_fraction = client_fraction
        self.aggregator = aggregator_cfg.get("aggregator", "fedavg")
        self.trim_ratio = float(aggregator_cfg.get("trim_ratio", 0.1))
        self.f = int(aggregator_cfg.get("f", 1))
        self.m = aggregator_cfg.get("m", None)
        if self.m is not None:
            self.m = int(self.m)

        self.attack_cfg = attack_cfg
        self.malicious_ids = malicious_ids

        self.defense_cfg = defense_cfg
        self.defense_enabled = bool(defense_cfg.get("enabled", False))
        self.k_exclude = int(defense_cfg.get("k_exclude", 1))
        self.kp = float(defense_cfg.get("Kp", 1.0))
        self.ki = float(defense_cfg.get("Ki", 0.0))
        self.kd = float(defense_cfg.get("Kd", 0.0))
        self.warmup_rounds = int(defense_cfg.get("warmup_rounds", 0))
        self.pid_state: dict[int, dict] = {}

        self.round_metrics: list[dict] = []
        self.agg_debug: list[dict] = []
        self.attack_debug: list[dict] = []
        self.defense_debug: list[dict] = []
        self._round_starts: dict[int, float] = {}
        self._train_loss_by_round: dict[int, float] = {}
        self._round_base_params: dict[int, list[np.ndarray]] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_starts[server_round] = time.perf_counter()
        self._round_base_params[server_round] = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def _aggregate_custom(self, server_round: int, results):
        ndarrays_by_client = [parameters_to_ndarrays(res.parameters) for _, res in results]
        vectors = [np.concatenate([a.ravel() for a in arrs]) for arrs in ndarrays_by_client]
        n_clients_total = len(vectors)

        if self.aggregator == "fedavg":
            weights = [res.num_examples for _, res in results]
            agg_vec = fedavg(vectors, weights=weights)
            n_used = n_clients_total
            n_selected = n_clients_total
        elif self.aggregator == "trimmed_mean":
            agg_vec = trimmed_mean(vectors, trim_ratio=self.trim_ratio)
            k = int(np.floor(self.trim_ratio * n_clients_total))
            n_used = max(1, n_clients_total - 2 * k)
            n_selected = n_used
        elif self.aggregator == "multi_krum":
            neighbor_count = n_clients_total - self.f - 2
            if neighbor_count <= 0:
                agg_vec = fedavg(vectors, weights=None)
                n_used = n_clients_total
                n_selected = n_clients_total
            else:
                m = self.m if self.m is not None else neighbor_count
                m = max(1, min(m, n_clients_total))
                agg_vec = multi_krum(vectors, f=self.f, m=m)
                n_used = n_clients_total
                n_selected = m
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")

        template = ndarrays_by_client[0]
        rebuilt, offset = [], 0
        for arr in template:
            size = int(np.prod(arr.shape))
            rebuilt_arr = agg_vec[offset : offset + size].reshape(arr.shape).astype(arr.dtype)
            rebuilt.append(rebuilt_arr)
            offset += size

        self.agg_debug.append(
            {
                "round": server_round,
                "aggregator": self.aggregator,
                "n_clients_total": n_clients_total,
                "n_clients_used": n_used,
                "n_selected": n_selected,
                "trim_ratio": self.trim_ratio if self.aggregator == "trimmed_mean" else "",
            }
        )
        return ndarrays_to_parameters(rebuilt)

    def _apply_pid_exclusion(self, server_round: int, results):
        if not results:
            return results, [], {}

        base = self._round_base_params.get(server_round)
        if base is None:
            return results, [], {}
        base_vec = np.concatenate([a.ravel() for a in base])

        client_ids = [int(res.metrics.get("client_id", idx)) for idx, (_, res) in enumerate(results)]
        updates = []
        for _, fit_res in results:
            vec = np.concatenate([a.ravel() for a in parameters_to_ndarrays(fit_res.parameters)])
            updates.append(vec - base_vec)
        updates_arr = np.asarray(updates)
        ref = np.median(updates_arr, axis=0)

        scores = {}
        for cid, upd in zip(client_ids, updates):
            error = float(np.linalg.norm(upd - ref))
            score, new_state = update_pid_score(error, self.pid_state.get(cid, {}), self.kp, self.ki, self.kd)
            self.pid_state[cid] = new_state
            scores[cid] = score

        excluded_ids = []
        if self.defense_enabled and server_round > self.warmup_rounds:
            k = max(0, min(self.k_exclude, len(results) - 1))
            excluded_ids = select_top_k_by_score(scores, k)

        kept = []
        for pair, cid in zip(results, client_ids):
            if cid not in excluded_ids:
                kept.append(pair)
        if not kept:
            kept = [results[0]]

        return kept, excluded_ids, scores

    def aggregate_fit(self, server_round, results, failures):
        total_examples, weighted_loss = 0, 0.0
        participating_ids = []
        for _, fit_res in results:
            n = fit_res.num_examples
            total_examples += n
            weighted_loss += n * float(fit_res.metrics.get("train_loss", 0.0))
            if "client_id" in fit_res.metrics:
                participating_ids.append(int(fit_res.metrics["client_id"]))
        self._train_loss_by_round[server_round] = weighted_loss / max(total_examples, 1)

        malicious_participating = sorted([cid for cid in participating_ids if cid in self.malicious_ids])
        self.attack_debug.append(
            {
                "round": server_round,
                "malicious_fraction": float(self.attack_cfg.get("malicious_fraction", 0.0)),
                "malicious_count": len(malicious_participating),
                "total_clients_sampled": len(participating_ids),
                "malicious_ids_participated": "|".join(str(cid) for cid in malicious_participating),
            }
        )

        filtered_results, excluded_ids, scores = self._apply_pid_exclusion(server_round, results)

        n_total = len(results)
        n_excluded = len(excluded_ids)
        malicious_in_round = set(malicious_participating)
        excluded_set = set(excluded_ids)
        if bool(self.attack_cfg.get("enabled", False)):
            tp = len(excluded_set & malicious_in_round)
            fp = len(excluded_set - malicious_in_round)
            fn = len(malicious_in_round - excluded_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            tp = fp = fn = ""
            precision = recall = ""

        score_str = "|".join(f"{cid}:{scores[cid]:.6f}" for cid in sorted(scores.keys()))
        self.defense_debug.append(
            {
                "round": server_round,
                "n_total": n_total,
                "n_excluded": n_excluded,
                "excluded_ids": "|".join(str(cid) for cid in sorted(excluded_ids)),
                "malicious_ids_in_round": "|".join(str(cid) for cid in sorted(malicious_in_round)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "scores": score_str,
            }
        )

        if not filtered_results:
            return None, {}
        params = self._aggregate_custom(server_round, filtered_results)
        return params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)
        total_examples, weighted_acc = 0, 0.0
        for _, eval_res in results:
            n = eval_res.num_examples
            total_examples += n
            weighted_acc += n * float(eval_res.metrics.get("val_acc", 0.0))
        val_acc = weighted_acc / max(total_examples, 1)

        time_sec = time.perf_counter() - self._round_starts.get(server_round, time.perf_counter())
        self.round_metrics.append(
            {
                "round": server_round,
                "client_fraction": round(self.client_fraction, 6),
                "train_loss": round(self._train_loss_by_round.get(server_round, 0.0), 6),
                "val_loss": round(float(agg_loss or 0.0), 6),
                "val_acc": round(val_acc, 6),
                "time_round_sec": round(time_sec, 6),
            }
        )
        if isinstance(agg_metrics, dict):
            agg_metrics["val_acc"] = val_acc
        return agg_loss, agg_metrics


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


def _write_defense_debug(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "round",
        "n_total",
        "n_excluded",
        "excluded_ids",
        "malicious_ids_in_round",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "scores",
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
) -> None:
    dataset_cfg = config.get("dataset", {})
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "fedavg_tiny",
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


def run_fedavg_tiny(config: dict, run_id: str, out_dir_results: str) -> tuple[list[dict], str]:
    """Run tiny FL with configurable robust aggregation; return metrics and plot tag."""
    _set_global_seed(int(config.get("seed", 42)))

    rounds = int(config.get("rounds", 2))
    num_clients = int(config.get("clients", 4))
    client_fraction = float(config.get("client_fraction", 1.0))
    fit_clients = max(1, int(np.ceil(num_clients * client_fraction)))
    server_cfg = config.get("server", {})
    aggregator = server_cfg.get("aggregator", "fedavg")

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
    )
    _write_agg_debug(results_root / "agg_debug.csv", strategy.agg_debug)
    _write_attack_debug(results_root / "attack_debug.csv", strategy.attack_debug)
    _write_defense_debug(results_root / "defense_debug.csv", strategy.defense_debug)

    return strategy.round_metrics, f"fedavg_tiny_{aggregator}"
