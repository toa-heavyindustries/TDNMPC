"""Minimal tests for LV feeder option and envelope script helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from coord.ti_env import apply_envelope_to_model, compute_envelope
from dso.network import build_ieee_european_lv_feeder
from opt.pyomo_dso import DSOParameters, build_dso_model
from utils.timewin import make_horizon


def test_build_lv_feeder_basic() -> None:
    feeder = build_ieee_european_lv_feeder("on_mid")
    # Basic structure
    assert feeder.feeder_type == "lv"
    assert feeder.root_bus in feeder.net.bus.index
    # Boundary flag present
    assert "is_boundary" in feeder.net.bus.columns


def test_compute_and_apply_envelope_to_dso_model(tmp_path: Path) -> None:
    # Fake history with horizon matrices (two scenarios)
    history = pd.DataFrame(
        {
            "dso_h": [
                np.array([[0.5, 0.0], [0.4, 0.1]]),
                np.array([[0.6, -0.1], [0.5, 0.2]]),
            ]
        }
    )
    env = compute_envelope(history, {"alpha": 1.0, "margin": 0.0})
    assert "lower" in env and "upper" in env
    assert env["lower"].shape == env["upper"].shape

    # Build a tiny DSO model with 2 buses and 2 steps
    horizon = make_horizon("2024-01-01 00:00", steps=2, dt_min=15)
    profiles = pd.DataFrame({"load": [1000.0, 1000.0], "pv": [0.0, 0.0]})
    sens = {"Rp": np.eye(2) * 0.01, "Rq": np.zeros((2, 2)), "vm_base": np.ones(2)}
    params = DSOParameters(sens=sens, profiles=profiles, horizon=horizon, penalty_voltage=1e3)
    model = build_dso_model(params)

    # Apply envelope to pg for bus indices [0,1]
    vars_map = {"pg": {"bus_indices": [0, 1], "penalty": 100.0, "dt_hours": horizon.dt_min / 60.0}}
    apply_envelope_to_model(model, env, vars_map)
    # Check constraints created
    # With soft penalty, variables p_env_pos/neg should exist
    assert hasattr(model, "p_env_pos") and hasattr(model, "p_env_neg")
    # Solve feasibility quickly with Ipopt if available or skip
    solver = pyo.SolverFactory("ipopt")
    if not solver.available():
        pytest.skip("Ipopt not available for quick feasibility check")
    res = solver.solve(model, tee=False)
    assert str(res.solver.status).lower() in {"ok", "warning"}

