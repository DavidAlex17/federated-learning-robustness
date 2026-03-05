from __future__ import annotations

import numpy as np

from experiments.methods.fedavg import _dirichlet_partition_indices, _partition_indices_iid


def _label_counts(partitions: list[list[int]], labels: np.ndarray) -> list[tuple[int, ...]]:
    out = []
    for part in partitions:
        cnt = [0] * 10
        for idx in part:
            cnt[int(labels[idx])] += 1
        out.append(tuple(cnt))
    return out


def test_dirichlet_partition_determinism() -> None:
    labels = np.array([i % 10 for i in range(200)], dtype=int)
    p1 = _dirichlet_partition_indices(labels=labels, num_clients=5, alpha=0.1, seed=42)
    p2 = _dirichlet_partition_indices(labels=labels, num_clients=5, alpha=0.1, seed=42)
    p3 = _dirichlet_partition_indices(labels=labels, num_clients=5, alpha=0.1, seed=43)

    assert _label_counts(p1, labels) == _label_counts(p2, labels)
    assert _label_counts(p1, labels) != _label_counts(p3, labels)


def test_iid_partition_basic_sanity() -> None:
    n = 200
    labels = np.array([i % 10 for i in range(n)], dtype=int)
    train_indices = np.arange(n)
    parts = _partition_indices_iid(train_indices=train_indices, num_clients=5, seed=1)
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1

    counts = _label_counts(parts, labels)
    # each client should see multiple labels in IID split
    assert all(sum(1 for c in client if c > 0) >= 5 for client in counts)
