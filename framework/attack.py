"""Attack injection utilities: sign-flip and malicious client selection."""

from __future__ import annotations

import numpy as np


def select_malicious_client_ids(num_clients: int, malicious_fraction: float, seed: int) -> set[int]:
    """Select a deterministic malicious client-id set."""
    malicious_count = int(np.floor(max(0.0, min(1.0, malicious_fraction)) * num_clients))
    rng = np.random.default_rng(seed)
    ids = rng.permutation(num_clients)[:malicious_count]
    return set(int(i) for i in ids)


def apply_signflip_attack(old_params: list[np.ndarray], new_params: list[np.ndarray], scale: float) -> list[np.ndarray]:
    """Apply sign-flip to parameter update: old + (-scale * (new - old))."""
    attacked = []
    for old, new in zip(old_params, new_params):
        update = new - old
        attacked_update = -float(scale) * update
        attacked.append(old + attacked_update)
    return attacked
