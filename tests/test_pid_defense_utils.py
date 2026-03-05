from __future__ import annotations

import numpy as np

from framework.defense import cosine_direction_error, select_top_k_by_score, update_pid_score


def test_update_pid_score_series() -> None:
    state = {}
    score1, state = update_pid_score(error=2.0, state=state, kp=1.0, ki=0.5, kd=0.1)
    score2, state = update_pid_score(error=3.0, state=state, kp=1.0, ki=0.5, kd=0.1)

    # round1: I=2, D=2 => 2 + 1 + 0.2 = 3.2
    assert abs(score1 - 3.2) < 1e-9
    # round2: I=5, D=1 => 3 + 2.5 + 0.1 = 5.6
    assert abs(score2 - 5.6) < 1e-9


def test_select_top_k_by_score() -> None:
    scores = {0: 0.2, 1: 1.5, 2: 1.5, 3: 0.1}
    # tie broken by smaller client_id first
    assert select_top_k_by_score(scores, k=2) == [1, 2]


def test_cosine_direction_error_extremes() -> None:
    v = np.array([1.0, 2.0, 3.0])
    e_same, cos_same = cosine_direction_error(v, v)
    e_opp, cos_opp = cosine_direction_error(-v, v)
    e_orth, cos_orth = cosine_direction_error(np.array([2.0, -1.0, 0.0]), np.array([1.0, 2.0, 0.0]))

    assert np.isclose(cos_same, 1.0)
    assert np.isclose(e_same, 0.0)

    assert np.isclose(cos_opp, -1.0)
    assert np.isclose(e_opp, 2.0)

    assert np.isclose(cos_orth, 0.0)
    assert np.isclose(e_orth, 1.0)
