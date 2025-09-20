"""Trajectory improvement (TI) envelope utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Envelope:
    """Stores smoothed lower/upper bounds for interface signals."""

    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    margin: float
    count: int = 0

    def as_dict(self) -> dict[str, list[float] | float | int]:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "alpha": self.alpha,
            "margin": self.margin,
            "count": self.count,
        }


def create_envelope(size: int, margin: float = 0.05, alpha: float = 0.2) -> Envelope:
    """Initialise an envelope with symmetric margins around zero."""

    if size <= 0:
        raise ValueError("size must be positive")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    lower = -np.ones(size) * margin
    upper = np.ones(size) * margin
    return Envelope(lower=lower, upper=upper, alpha=float(alpha), margin=float(margin))


def update_envelope(env: Envelope, values: np.ndarray) -> Envelope:
    """Update envelope bounds using exponential smoothing towards new extrema."""

    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != env.lower.shape[0]:
        raise ValueError("value dimension mismatch")

    target_lower = np.minimum(env.lower, arr - env.margin)
    target_upper = np.maximum(env.upper, arr + env.margin)

    env.lower = (1.0 - env.alpha) * env.lower + env.alpha * target_lower
    env.upper = (1.0 - env.alpha) * env.upper + env.alpha * target_upper

    env.lower = np.minimum(env.lower, arr - env.margin)
    env.upper = np.maximum(env.upper, arr + env.margin)
    env.count += 1
    return env


def violation_stats(env: Envelope, values: np.ndarray) -> dict[str, float]:
    """Compute violation metrics for a batch of values."""

    arr = np.asarray(values, dtype=float)
    if arr.shape[-1] != env.lower.shape[0]:
        raise ValueError("value dimension mismatch")

    violations_low = np.maximum(env.lower - arr, 0.0)
    violations_high = np.maximum(arr - env.upper, 0.0)
    total_violation = violations_low + violations_high

    max_violation = float(total_violation.max()) if total_violation.size else 0.0
    l2_violation = float(np.linalg.norm(total_violation.ravel()))
    rate = float(np.mean(total_violation > 0)) if total_violation.size else 0.0

    return {
        "max": max_violation,
        "l2": l2_violation,
        "rate": rate,
    }

