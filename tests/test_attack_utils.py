from __future__ import annotations

import numpy as np

from framework.attack import apply_signflip_attack, select_malicious_client_ids


def test_select_malicious_client_ids_is_deterministic() -> None:
    a = select_malicious_client_ids(num_clients=10, malicious_fraction=0.3, seed=7)
    b = select_malicious_client_ids(num_clients=10, malicious_fraction=0.3, seed=7)
    c = select_malicious_client_ids(num_clients=10, malicious_fraction=0.3, seed=8)
    assert a == b
    assert len(a) == 3
    assert a != c


def test_apply_signflip_attack_transform() -> None:
    old_params = [np.array([1.0, 2.0]), np.array([3.0])]
    new_params = [np.array([2.0, 4.0]), np.array([5.0])]
    attacked = apply_signflip_attack(old_params, new_params, scale=1.0)

    # update: [1,2], [2] -> attacked update [-1,-2],[-2]
    assert np.allclose(attacked[0], np.array([0.0, 0.0]))
    assert np.allclose(attacked[1], np.array([1.0]))
