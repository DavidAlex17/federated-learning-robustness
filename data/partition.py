"""IID and Dirichlet label-skew partitioning of dataset indices."""

from __future__ import annotations

import numpy as np


def _partition_indices_iid(train_indices: np.ndarray, num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(train_indices, copy=True)
    rng.shuffle(shuffled)
    return [arr.tolist() for arr in np.array_split(shuffled, num_clients)]


def _dirichlet_partition_indices(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[list[int]]:
    """Deterministic Dirichlet label-skew partitioning with repair for empty clients."""
    labels = np.asarray(labels)
    n = labels.shape[0]
    rng = np.random.default_rng(seed)
    per_client: list[list[int]] = [[] for _ in range(num_clients)]

    classes = np.unique(labels)
    for c in classes:
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        if len(class_indices) == 0:
            continue
        probs = rng.dirichlet(np.full(num_clients, alpha))
        counts = rng.multinomial(len(class_indices), probs)
        start = 0
        for cid, cnt in enumerate(counts):
            if cnt > 0:
                per_client[cid].extend(class_indices[start : start + cnt].tolist())
            start += cnt

    # repair: ensure each client has at least one sample when possible
    if n >= num_clients:
        for cid in range(num_clients):
            if len(per_client[cid]) == 0:
                donor = max(range(num_clients), key=lambda i: len(per_client[i]))
                if len(per_client[donor]) > 1:
                    per_client[cid].append(per_client[donor].pop())

    return per_client
