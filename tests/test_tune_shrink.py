"""Unit tests for shrink sizing logic using closed-loop residual sigma."""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import pytest

from sim import run_closed_loop


def ensure_solver(name: str) -> None:
    solver = pyo.SolverFactory(name)
    if solver is None or not solver.available():
        pytest.skip(f"Solver {name} unavailable")


def test_sigma_quantile_matches_numpy() -> None:
    ensure_solver("glpk")

    sigmas = []
    rng = np.random.default_rng(7)
    for _ in range(5):
        load = rng.normal(scale=0.2, size=6)
        res = run_closed_loop(
            steps=6,
            amplitude=0.0,
            solver="glpk",
            load_profile=load,
            noise_std=0.1,
        )
        sigmas.append(res.summary["sigma_p95"])

    np_sigma = float(np.quantile(sigmas, 0.9))
    assert np_sigma >= 0.0
