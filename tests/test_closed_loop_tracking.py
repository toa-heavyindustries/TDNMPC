"""Integration tests for the closed-loop multi-step simulation."""

from __future__ import annotations

import pyomo.environ as pyo
import pytest

from sim import run_closed_loop


def ensure_solver(name: str) -> None:
    solver = pyo.SolverFactory(name)
    if solver is None or not solver.available():
        pytest.skip(f"Solver {name} unavailable")


def test_run_closed_loop_summary() -> None:
    ensure_solver("glpk")
    load_profile = [12.0, 12.5, 13.0, 11.8]
    result = run_closed_loop(steps=4, amplitude=0.0, solver="glpk", load_profile=load_profile, dt_min=10.0)

    history = result.history
    assert not history.empty
    assert set(history.columns) >= {"step", "bus", "p_target", "p_tso", "residual", "p_request", "soc"}

    summary = result.summary
    assert summary["max_abs_residual"] <= 1e-3
    assert 0.9 <= summary["min_voltage"] <= 1.05
    assert summary["max_voltage"] <= 1.05
    assert summary["soc_range"] >= 0.0
    assert summary["voltage_violations"] == 0.0
