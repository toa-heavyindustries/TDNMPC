"""Tests for time horizon and rolling window utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.timewin import Horizon, align_profiles, make_horizon, rolling_windows


def test_make_horizon_properties() -> None:
    horizon = make_horizon("2024-01-01 00:00", steps=4, dt_min=15)
    assert isinstance(horizon, Horizon)
    assert horizon.steps == 4
    assert horizon.dt_min == 15
    assert horizon.end == pd.Timestamp("2024-01-01 01:00")


def test_rolling_windows_no_overlap() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="15min")
    windows = rolling_windows(idx, steps=4)
    assert len(windows) == 3
    for w in windows:
        assert len(w) == 4
    for first, second in zip(windows, windows[1:]):
        assert first[-1] < second[0]


def test_align_profiles_fills_missing() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="1H")
    series = pd.Series([1, 2, 3], index=idx[:3])
    df = align_profiles(idx, {"load": series})
    assert df.shape == (6, 1)
    assert df.iloc[0, 0] == pytest.approx(1.0)
    assert df.iloc[-1, 0] == pytest.approx(3.0)
    assert df.index.name == "time"

