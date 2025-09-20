"""Tests for the ADMM coordination utilities."""

from __future__ import annotations

import numpy as np
import pytest

from coord.admm import ADMMConfig, run_admm


def quadratic_updates(target: np.ndarray, rho: float):
    def prox(v: np.ndarray) -> np.ndarray:
        return (target + rho * v) / (1.0 + rho)

    return prox


def test_run_admm_converges_to_consensus():
    size = 3
    rho = 1.2
    target_tso = np.array([1.0, -2.0, 0.5])
    target_dso = np.array([-1.0, 1.0, 2.0])

    tso_step = quadratic_updates(target_tso, rho)
    dso_step = quadratic_updates(target_dso, rho)

    cfg = ADMMConfig(size=size, rho=rho, max_iters=200, tol_primal=1e-6, tol_dual=1e-6)
    result = run_admm(tso_step, dso_step, cfg)

    assert result["converged"]
    x = result["x"]
    z = result["z"]
    avg = 0.5 * (target_tso + target_dso)
    assert np.allclose(x, avg, atol=1e-5)
    assert np.allclose(z, avg, atol=1e-5)

    history = result["history"]
    assert history[0]["primal_residual"] > history[-1]["primal_residual"]
    assert history[-1]["primal_residual"] < 1e-6
    assert history[-1]["dual_residual"] < 1e-6


def test_admm_invalid_size_raises():
    cfg = ADMMConfig(size=0)
    with pytest.raises(ValueError):
        run_admm(lambda v: v, lambda v: v, cfg)

