"""Evaluation metrics for NMPC experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def voltage_violation(df: pd.DataFrame, vmin: float = 0.95, vmax: float = 1.05) -> float:
    """Return the fraction of samples violating voltage bounds."""

    mask_low = df < vmin
    mask_high = df > vmax
    violations = mask_low | mask_high
    total = violations.size
    if total == 0:
        return 0.0
    return float(violations.sum() / total)


def energy_cost(costs: pd.Series) -> float:
    """Sum energy costs with NaNs treated as zero."""

    return float(costs.fillna(0.0).sum())


def coupling_rmse(residuals: pd.Series) -> float:
    """Compute the root-mean-square error for coupling residual series."""

    arr = residuals.fillna(0.0).to_numpy()
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr**2)))

