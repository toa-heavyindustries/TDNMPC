"""Integration test for envelope-constrained TSO/DSO tracking step."""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import pytest

from dso.network import build_cigre_feeder
from interface import apply_reference, build_box_envelope, envelopes_to_bounds, measure_boundary
from opt.pyomo_tso import TSOParameters, build_tso_model, extract_solution, solve_tso_model
from tso.network import build_tso_case, mark_boundary_buses


def ensure_solver(name: str) -> None:
    solver = pyo.SolverFactory(name)
    if solver is None or not solver.available():
        pytest.skip(f"Solver {name} unavailable")


def test_envelope_constrained_tracking_step() -> None:
    ensure_solver("glpk")

    case = build_tso_case(8)
    boundary_ids = [1, 3, 5]
    case = mark_boundary_buses(case, boundary_ids)

    feeders = [build_cigre_feeder("mv", target_peak_mw=20.0) for _ in boundary_ids]
    envelopes = []
    responses: dict[int, float] = {}

    for bus, feeder in zip(boundary_ids, feeders):
        env = build_box_envelope(feeder, p_margin=5.0, q_margin=3.0)
        env.metadata["boundary_bus"] = bus
        envelopes.append(env)

        p_base, q_base, _ = measure_boundary(feeder.net)
        target_p = max(env.p_min + 0.5, min(env.p_max - 0.5, p_base - 1.0))
        tracking = apply_reference(feeder.net, target_p, q_base)
        responses[bus] = tracking.p_mw

    lower, upper = envelopes_to_bounds(envelopes, case["boundary"])
    boundary_targets = np.array([responses[int(bus)] for bus in case["boundary"]], dtype=float)

    params = TSOParameters(
        admittance=case["admittance"],
        injections=case["injections"],
        boundary=case["boundary"],
        boundary_targets=boundary_targets,
        rho=1.0,
        cost_coeff=10.0,
        lower_bounds=lower,
        upper_bounds=upper,
    )

    model = build_tso_model(params)
    solve_tso_model(model, solver="glpk")
    result = extract_solution(model, params)

    flows = result.flows.loc[case["boundary"]]
    assert np.all(flows.to_numpy() <= upper + 1e-6)
    assert np.all(flows.to_numpy() >= lower - 1e-6)

    for idx, bus in enumerate(case["boundary"]):
        assert flows.iloc[idx] == pytest.approx(boundary_targets[idx], abs=1e-3)
