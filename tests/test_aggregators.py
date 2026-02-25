from __future__ import annotations

import numpy as np

from framework.aggregators import fedavg, multi_krum, trimmed_mean


def test_fedavg_mean_and_weighted_mean() -> None:
    vectors = [np.array([1.0, 3.0]), np.array([3.0, 5.0])]
    assert np.allclose(fedavg(vectors), np.array([2.0, 4.0]))
    assert np.allclose(fedavg(vectors, weights=[1.0, 3.0]), np.array([2.5, 4.5]))


def test_trimmed_mean_trims_outliers() -> None:
    vectors = [
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([2.0, 2.0]),
        np.array([2.0, 2.0]),
        np.array([100.0, 100.0]),
    ]
    out = trimmed_mean(vectors, trim_ratio=0.2)
    assert np.allclose(out, np.array([2.0, 2.0]))


def test_multi_krum_selects_clustered_vectors() -> None:
    vectors = [
        np.array([0.00, 0.00]),
        np.array([0.10, -0.10]),
        np.array([-0.10, 0.10]),
        np.array([10.0, 10.0]),
    ]
    out = multi_krum(vectors, f=1, m=2)
    expected = np.array([0.05, -0.05])
    assert np.allclose(out, expected)
