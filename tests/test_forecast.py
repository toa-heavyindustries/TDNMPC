"""Tests for baseline forecasting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim.forecast import forecast_naive, forecast_sma


def test_forecast_naive_repeats_last_value():
    series = pd.Series([1.0, 2.0, 3.5])
    out = forecast_naive(series, horizon=3)
    assert np.allclose(out, np.array([3.5, 3.5, 3.5]))


def test_forecast_naive_raises_on_empty():
    with pytest.raises(ValueError):
        forecast_naive(pd.Series(dtype=float), horizon=3)


def test_forecast_sma_averages_last_window():
    series = pd.Series([1, 2, 3, 4, 5])
    out = forecast_sma(series, horizon=2, w=3)
    assert np.allclose(out, np.array([4.0, 4.0]))


def test_forecast_sma_handles_all_nan():
    series = pd.Series([np.nan, np.nan])
    out = forecast_sma(series, horizon=2, w=3)
    assert np.allclose(out, np.zeros(2))

