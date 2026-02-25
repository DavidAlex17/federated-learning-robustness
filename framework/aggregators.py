"""Robust aggregation baselines on flattened NumPy vectors."""

from __future__ import annotations

import numpy as np


def _validate_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("vectors must be a non-empty list")
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("vectors must be 2D: (n_clients, n_params)")
    return arr


def fedavg(vectors: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Compute weighted or unweighted average across client vectors."""
    arr = _validate_vectors(vectors)
    if weights is None:
        return arr.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != arr.shape[0]:
        raise ValueError("weights must be 1D with same length as vectors")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("sum of weights must be > 0")
    return np.average(arr, axis=0, weights=w)


def trimmed_mean(vectors: list[np.ndarray], trim_ratio: float) -> np.ndarray:
    """Coordinate-wise trimmed mean removing extremes per coordinate."""
    arr = _validate_vectors(vectors)
    n = arr.shape[0]
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must satisfy 0 <= trim_ratio < 0.5")
    k = int(np.floor(trim_ratio * n))
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    sorted_arr = np.sort(arr, axis=0)
    kept = sorted_arr[k : n - k] if k > 0 else sorted_arr
    if kept.shape[0] == 0:
        kept = sorted_arr
    return kept.mean(axis=0)


def multi_krum(vectors: list[np.ndarray], f: int, m: int | None = None) -> np.ndarray:
    """Compute Multi-Krum aggregate (simple mean over selected clients).

    Score(i) = sum of smallest (n - f - 2) squared distances from client i.
    Select m clients with smallest scores (default m = n - f - 2).
    """
    arr = _validate_vectors(vectors)
    n = arr.shape[0]
    if f < 0:
        raise ValueError("f must be >= 0")
    if n < 3:
        raise ValueError("multi_krum requires at least 3 vectors")
    neighbor_count = n - f - 2
    if neighbor_count <= 0:
        raise ValueError("invalid parameters: require n - f - 2 > 0")

    if m is None:
        m = neighbor_count
    if m <= 0 or m > n:
        raise ValueError("m must satisfy 1 <= m <= n")

    diffs = arr[:, None, :] - arr[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)

    scores = []
    for i in range(n):
        row = np.delete(dist2[i], i)
        smallest = np.partition(row, neighbor_count - 1)[:neighbor_count]
        scores.append(float(np.sum(smallest)))
    scores_arr = np.asarray(scores)
    selected = np.argsort(scores_arr)[:m]
    return arr[selected].mean(axis=0)
