"""MNIST data loading, capping, and per-client partitioned subsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms

from data.partition import _dirichlet_partition_indices, _partition_indices_iid


@dataclass
class PartitionBundle:
    train_parts: list[Subset]
    test_set: Subset
    dataset_used: str


def _cap_dataset(ds, cap: int, seed: int) -> Subset:
    indices = np.arange(len(ds))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    indices = indices[: min(cap, len(indices))]
    return Subset(ds, indices.tolist())


def _build_partitioned_data(config: dict) -> PartitionBundle:
    dataset_cfg = config.get("dataset", {})
    partition_cfg = config.get("partition", {})
    train_cap = int(dataset_cfg.get("train_samples_cap", 1000))
    test_cap = int(dataset_cfg.get("test_samples_cap", 200))
    num_clients = int(config.get("clients", 4))
    seed = int(config.get("seed", 42))
    partition_type = str(partition_cfg.get("type", "iid")).lower()
    partition_alpha = float(partition_cfg.get("alpha", 0.1))

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

    train_indices = np.array(train_subset.indices)
    labels = np.array([int(train_raw[idx][1]) for idx in train_indices], dtype=int)

    if partition_type == "dirichlet":
        local_lists = _dirichlet_partition_indices(labels=labels, num_clients=num_clients, alpha=partition_alpha, seed=seed)
    else:
        local_lists = _partition_indices_iid(train_indices=np.arange(len(train_indices)), num_clients=num_clients, seed=seed)

    train_parts = []
    for local in local_lists:
        mapped = train_indices[np.asarray(local, dtype=int)].tolist() if local else []
        train_parts.append(Subset(train_raw, mapped))

    return PartitionBundle(train_parts=train_parts, test_set=test_subset, dataset_used=dataset_used)
