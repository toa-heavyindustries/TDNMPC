"""Tests for the TSO DC power flow utilities."""

from __future__ import annotations

import numpy as np

from tso.network import build_tso_case, dc_power_flow, mark_boundary_buses


def test_build_tso_case_shapes() -> None:
    case = build_tso_case(10)
    assert case["buses"].shape == (10,)
    assert case["admittance"].shape == (10, 10)
    assert np.isclose(case["injections"].sum(), 0.0)


def test_mark_boundary_buses_sets_sorted_unique() -> None:
    case = build_tso_case(5)
    updated = mark_boundary_buses(case, [3, 1, 3])
    assert updated["boundary"].tolist() == [1, 3]
    assert case["boundary"].size == 0  # original case unchanged


def test_dc_power_flow_residual_small() -> None:
    case = build_tso_case(8)
    sol = dc_power_flow(case)
    residual = case["admittance"] @ sol["theta"] - case["injections"]
    assert np.linalg.norm(residual, ord=np.inf) < 1e-6

