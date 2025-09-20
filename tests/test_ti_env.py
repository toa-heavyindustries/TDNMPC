"""Tests for the TI envelope utilities."""

from __future__ import annotations

import numpy as np

from coord.ti_env import Envelope, create_envelope, update_envelope, violation_stats


def test_create_envelope_basic():
    env = create_envelope(size=3, margin=0.1, alpha=0.5)
    assert isinstance(env, Envelope)
    assert np.allclose(env.lower, -0.1)
    assert np.allclose(env.upper, 0.1)
    assert env.alpha == 0.5
    assert env.count == 0


def test_update_envelope_shrinks_towards_values():
    env = create_envelope(size=2, margin=0.05, alpha=0.5)
    values = np.array([0.2, -0.3])
    update_envelope(env, values)
    assert env.count == 1
    # Envelope should expand to include values with margin
    assert env.upper[0] >= values[0]
    assert env.lower[1] <= values[1]


def test_violation_stats_detects_outliers():
    env = create_envelope(size=2, margin=0.05, alpha=0.5)
    env.lower = np.array([-0.1, -0.1])
    env.upper = np.array([0.1, 0.1])
    values = np.array([[0.0, 0.2], [0.15, -0.2]])
    stats = violation_stats(env, values)
    assert stats["rate"] > 0
    assert stats["max"] > 0.09
    assert stats["l2"] > 0

