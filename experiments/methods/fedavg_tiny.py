"""Tiny FedAvg backend using Flower + PyTorch on capped MNIST."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class PartitionBundle:
    train_parts: list[Subset]
    test_set: Subset


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset_used = "mnist"
    try:
        train_raw = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_raw = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    except RuntimeError:
        dataset_used = "fakedata"
        train_raw = datasets.FakeData(
            size=max(train_cap, 1),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed,
        )
        test_raw = datasets.FakeData(
            size=max(test_cap, 1),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed + 1,
        )

    print(f"dataset={dataset_used}")

    train_subset = _cap_dataset(train_raw, train_cap, seed=seed)
    test_subset = _cap_dataset(test_raw, test_cap, seed=seed + 1)

    rng = np.random.default_rng(seed)
    train_indices = np.array(train_subset.indices)
    rng.shuffle(train_indices)
    split_indices = np.array_split(train_indices, num_clients)
    train_parts = [Subset(train_raw, idxs.tolist()) for idxs in split_indices]

    return PartitionBundle(train_parts=train_parts, test_set=test_subset)


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
    loss_sum = 0.0
    n = 0
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
    loss_sum = 0.0
    correct = 0
    n = 0
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
    ) -> None:
        self.cid = cid
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
        _set_params(self.model, parameters)
        losses = []
        for _ in range(self.local_epochs):
            losses.append(_train_one_epoch(self.model, self.train_loader, self.lr, self.device))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        return _get_params(self.model), len(self.train_loader.dataset), {"train_loss": mean_loss}

    def evaluate(self, parameters, config):
        _set_params(self.model, parameters)
        val_loss, val_acc = _evaluate(self.model, self.test_loader, self.device)
        return float(val_loss), len(self.test_loader.dataset), {"val_acc": float(val_acc)}


class MetricsFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, client_fraction: float, **kwargs):
        super().__init__(**kwargs)
        self.client_fraction = client_fraction
        self.round_metrics: list[dict] = []
        self._round_starts: dict[int, float] = {}
        self._train_loss_by_round: dict[int, float] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_starts[server_round] = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        total_examples = 0
        weighted_loss = 0.0
        for _, fit_res in results:
            n = fit_res.num_examples
            total_examples += n
            weighted_loss += n * float(fit_res.metrics.get("train_loss", 0.0))
        self._train_loss_by_round[server_round] = weighted_loss / max(total_examples, 1)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)

        total_examples = 0
        weighted_acc = 0.0
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


def run_fedavg_tiny(config: dict) -> list[dict]:
    """Run tiny FedAvg and return metrics rows in shared schema."""
    _set_global_seed(int(config.get("seed", 42)))

    rounds = int(config.get("rounds", 2))
    num_clients = int(config.get("clients", 4))
    client_fraction = float(config.get("client_fraction", 1.0))
    fit_clients = max(1, int(np.ceil(num_clients * client_fraction)))

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
        )
        if hasattr(client, "to_client"):
            return client.to_client()
        return client

    strategy = MetricsFedAvg(
        client_fraction=client_fraction,
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

    return strategy.round_metrics
