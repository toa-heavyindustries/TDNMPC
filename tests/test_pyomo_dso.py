"""Tests for the Pyomo-based DSO optimisation model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from opt.pyomo_dso import DSOParameters, build_dso_model, extract_solution, solve_dso_model
from utils.timewin import make_horizon


@pytest.fixture()
def simple_params():
    horizon = make_horizon("2024-01-01 00:00", steps=3, dt_min=15)
    profiles = pd.DataFrame(
        {
            "load": [900.0, 950.0, 980.0],
            "pv": [100.0, 120.0, 140.0],
        }
    )
    sens = {
        "Rp": np.eye(2) * 0.01,
        "Rq": np.zeros((2, 2)),
        "vm_base": np.ones(2),
    }
    return DSOParameters(sens=sens, profiles=profiles, horizon=horizon, penalty_voltage=1e3)


def test_build_dso_model_variables(simple_params) -> None:
    model = build_dso_model(simple_params)
    assert isinstance(model, pyo.ConcreteModel)
    assert len(list(model.B)) == 2
    assert len(list(model.T)) == 3


def test_solve_and_extract(simple_params) -> None:
    model = build_dso_model(simple_params)
    solver = "gurobi" if pyo.SolverFactory("gurobi").available() else "ipopt"
    try:
        solve_dso_model(model, solver=solver, options={"time_limit_seconds": 1})
    except RuntimeError as exc:
        if "available" in str(exc):
            pytest.skip("Required solver unavailable")
        raise

    result = extract_solution(model, simple_params)
    assert result.p_injections.shape == (3, 2)
    assert result.voltage.min().min() >= simple_params.vmin - 1e-3
    assert result.voltage.max().max() <= simple_params.vmax + 1e-3
    assert result.objective >= 0.0
