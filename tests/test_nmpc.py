"""Tests for the NMPC controller wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from coord.admm import ADMMConfig
from nmpc.controller import NMPCConfig, NMPCController


@pytest.fixture()
def simple_controller():
    size = 2
    rho = 1.0
    admm_cfg = ADMMConfig(size=size, rho=rho, max_iters=100, tol_primal=1e-6, tol_dual=1e-6)

    target_tso = np.array([1.0, -0.5])
    target_dso = np.array([0.6, -0.2])

    def tso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        value = (target_tso + rho * v) / (1.0 + rho)
        return value, {"theta": value}

    def dso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        value = (target_dso + rho * v) / (1.0 + rho)
        return value, {"control": value}

    cfg = NMPCConfig(
        size=size,
        admm=admm_cfg,
        tso_solver=tso_solver,
        dso_solver=dso_solver,
        envelope_margin=0.05,
        envelope_alpha=0.5,
    )
    return NMPCController(cfg)


def test_nmpc_run_step_converges(simple_controller: NMPCController) -> None:
    result = simple_controller.run_step()
    avg = 0.5 * (np.array([1.0, -0.5]) + np.array([0.6, -0.2]))
    assert np.allclose(result.tso_vector, avg, atol=1e-5)
    assert np.allclose(result.dso_vector, avg, atol=1e-5)
    assert result.residuals["max"] < 1e-5
    assert result.envelope.count == 1


def test_nmpc_initial_guess_validation(simple_controller: NMPCController) -> None:
    with pytest.raises(ValueError):
        simple_controller.run_step(initial_guess=np.array([1.0]))

