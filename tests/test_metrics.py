"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.metrics import coupling_rmse, energy_cost, voltage_violation


def test_voltage_violation_fraction():
    df = pd.DataFrame({"a": [0.94, 0.96, 1.06, 1.02]})
    fraction = voltage_violation(df)
    assert fraction == pytest.approx(0.5)


def test_energy_cost_sum():
    series = pd.Series([10.0, np.nan, 5.0])
    assert energy_cost(series) == pytest.approx(15.0)


def test_coupling_rmse_zero_for_empty():
    assert coupling_rmse(pd.Series(dtype=float)) == 0.0


def test_coupling_rmse_handles_values():
    series = pd.Series([1.0, -1.0, 1.0, -1.0])
    assert coupling_rmse(series) == pytest.approx(1.0)

