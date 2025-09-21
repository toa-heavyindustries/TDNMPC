"""Tests for the Pyomo-based TSO optimisation module."""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import pytest

from opt.pyomo_tso import TSOParameters, build_tso_model, extract_solution, solve_tso_model


def make_case():
    admittance = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    injections = np.array([-40.0, 10.0, 30.0])
    boundary = np.array([2])
    boundary_targets = np.array([injections[2]])
    lower_bounds = np.array([20.0])
    upper_bounds = np.array([40.0])
    params = TSOParameters(
        admittance=admittance,
        injections=injections,
        boundary=boundary,
        boundary_targets=boundary_targets,
        rho=1.0,
        cost_coeff=10.0,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    return params


def test_build_tso_model_sets_boundary_and_internal():
    params = make_case()
    model = build_tso_model(params)
    assert isinstance(model, pyo.ConcreteModel)
    assert set(model.boundary) == {2}
    assert set(model.internal) == {0, 1}


def test_solve_and_extract_tso():
    params = make_case()
    model = build_tso_model(params)
    # Prefer gurobi; fall back to ipopt if unavailable
    solver = "gurobi" if pyo.SolverFactory("gurobi").available() else "ipopt"
    try:
        solve_tso_model(model, solver=solver, options={"time_limit_seconds": 1})
    except RuntimeError as exc:
        if "available" in str(exc):
            pytest.skip("Required solver unavailable")
        raise

    result = extract_solution(model, params)
    # Boundary flow equals net injection on boundary bus
    assert result.flows.loc[2] == pytest.approx(params.injections[2], abs=1e-6)
    # Internal adjustments should be finite
    assert np.isfinite(result.adjustments.loc[0])
    assert np.isfinite(result.adjustments.loc[1])

    # Power balance residuals for internal buses close to zero
    residual = result.flows.loc[[0, 1]] - (
        params.injections[[0, 1]] + result.adjustments.loc[[0, 1]].to_numpy()
    )
    assert np.linalg.norm(residual.values, ord=np.inf) < 1e-6
