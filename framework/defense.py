"""PID-based anomaly scoring for Byzantine-robust defense."""

from __future__ import annotations

import numpy as np


def cosine_direction_error(update: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """Return (error, cosine) where error = 1 - cosine, with safe zero-norm handling."""
    norm_u = float(np.linalg.norm(update))
    norm_r = float(np.linalg.norm(reference))
    denom = norm_u * norm_r + eps
    if norm_u <= eps or norm_r <= eps:
        return 0.0, 1.0
    cosine = float(np.dot(update, reference) / denom)
    if not np.isfinite(cosine):
        return 0.0, 1.0
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine, cosine


def update_pid_score(
    error: float,
    state: dict,
    kp: float,
    ki: float,
    kd: float,
    integral_decay: float = 1.0,
) -> tuple[float, dict]:
    """Update PID state from a scalar error and return (score, new_state).

    integral_decay < 1.0 applies exponential decay to the accumulated integral
    (leaky integrator), so transient noise in early rounds does not permanently
    penalise an otherwise-honest client.  decay=1.0 is the original behaviour.
    """
    integral = integral_decay * float(state.get("integral", 0.0)) + float(error)
    prev_error = float(state.get("prev_error", 0.0))
    derivative = float(error) - prev_error
    score = kp * float(error) + ki * integral + kd * derivative
    return float(score), {"integral": integral, "prev_error": float(error)}


def select_top_k_by_score(scores: dict[int, float], k: int) -> list[int]:
    """Select top-k client ids by descending score, with deterministic tie-break by client id."""
    if k <= 0:
        return []
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in ordered[: min(k, len(ordered))]]
