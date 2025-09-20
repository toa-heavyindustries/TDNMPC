"""Baseline forecasting utilities for load/PV/price signals."""

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

