"""Forecasting utilities for load/PV/price signals.

Includes:
- Naive and SMA baselines
- AR(1)-driven multi-scenario sampling with relative or additive error models
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_naive(series: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value across the forecast horizon."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if series.empty:
        raise ValueError("series must contain at least one value")

    last = float(series.dropna().iloc[-1]) if series.dropna().size else 0.0
    return np.full(horizon, last, dtype=float)


def forecast_sma(series: pd.Series, horizon: int, w: int = 6) -> np.ndarray:
    """Simple moving average forecast using the last ``w`` samples."""

    if w <= 0:
        raise ValueError("window must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if series.empty:
        raise ValueError("series must contain at least one value")

    clean = series.dropna()
    if clean.size == 0:
        return np.zeros(horizon, dtype=float)

    window = clean.iloc[-min(w, clean.size) :]
    mean_value = float(window.mean())
    return np.full(horizon, mean_value, dtype=float)


def sample_forecast(
    truth: pd.Series,
    sigma: float,
    rho: float,
    horizon: int,
    n: int = 1,
    mode: str = "relative",
    clamp_nonneg: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Generate ``n`` AR(1)-correlated forecast scenarios over a horizon.

    Parameters
    ----------
    truth:
        Ground-truth series for the target horizon. Must contain at least ``horizon`` points.
    sigma:
        Error scale. If ``mode='relative'``, interpreted as relative standard deviation (e.g. 0.05).
        If ``mode='additive'``, interpreted in absolute units of ``truth``.
    rho:
        AR(1) autocorrelation coefficient in [-0.99, 0.99].
    horizon:
        Number of forecast steps to generate.
    n:
        Number of scenarios to sample.
    mode:
        'relative' for multiplicative errors around truth, 'additive' for additive errors.
    clamp_nonneg:
        Clamp negative values to zero (useful for PV forecasts).
    seed:
        Optional seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n, horizon) with sampled forecast trajectories.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    if not (-0.99 <= rho <= 0.99):
        raise ValueError("rho must be in [-0.99, 0.99]")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    base = truth.dropna().to_numpy()
    if base.size < horizon:
        raise ValueError("truth series shorter than requested horizon")
    base = base[:horizon].astype(float)

    rng = np.random.default_rng(seed)
    # Innovations for AR(1)
    eps = rng.standard_normal(size=(n, horizon))
    e = np.zeros((n, horizon), dtype=float)
    # Initialize first error state with stationary variance sigma^2 for AR(1)
    if mode == "relative":
        # relative error multiplier around 1.0: (1 + e)
        scale = sigma
    elif mode == "additive":
        scale = sigma
    else:
        raise ValueError("mode must be 'relative' or 'additive'")

    var0 = scale**2 / max(1e-12, (1 - rho**2)) if abs(rho) < 1 else scale**2
    e[:, 0] = rng.normal(loc=0.0, scale=np.sqrt(var0), size=n)
    for t in range(1, horizon):
        e[:, t] = rho * e[:, t - 1] + scale * eps[:, t]

    if mode == "relative":
        forecasts = (1.0 + e) * base[None, :]
    else:  # additive
        forecasts = base[None, :] + e

    if clamp_nonneg:
        np.maximum(forecasts, 0.0, out=forecasts)

    return forecasts
