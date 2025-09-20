"""Utilities for constructing and solving simplified DC transmission networks."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_tso_case(n_bus: int = 30) -> dict[str, Any]:
    """Create a deterministic DC power flow test case."""

    if n_bus < 2:
        raise ValueError("TSO case requires at least two buses")

    buses = np.arange(n_bus, dtype=int)
    slack = int(buses[0])

    rng = np.random.default_rng(123)

    adjacency = np.zeros((n_bus, n_bus), dtype=float)
    for idx in range(n_bus - 1):
        weight = float(rng.uniform(0.1, 1.0))
        adjacency[idx, idx + 1] = adjacency[idx + 1, idx] = weight

    extra_edges = max(n_bus // 2, 1)
    for _ in range(extra_edges):
        i, j = rng.choice(n_bus, size=2, replace=False)
        weight = float(rng.uniform(0.05, 0.5))
        adjacency[i, j] = adjacency[j, i] = weight

    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency

    injections = rng.normal(0.0, 50.0, size=n_bus)
    injections[slack] = -float(np.sum(injections[buses != slack]))

    return {
        "buses": buses,
        "slack": slack,
        "admittance": laplacian,
        "injections": injections,
        "boundary": np.array([], dtype=int),
    }


def mark_boundary_buses(case: dict[str, Any], bus_ids: list[int]) -> dict[str, Any]:
    """Return a copy of ``case`` with boundary buses marked."""

    buses = set(case["buses"].tolist())
    boundary = []
    for bus in bus_ids:
        if bus not in buses:
            raise ValueError(f"Invalid bus id: {bus}")
        boundary.append(int(bus))

    new_case = case.copy()
    new_case["boundary"] = np.array(sorted(set(boundary)), dtype=int)
    return new_case


def dc_power_flow(case: dict[str, Any], injections: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Solve the DC power flow for the given case."""

    theta = _solve_dc(case, injections)
    flows = case["admittance"] @ theta
    return {"theta": theta, "flows": flows}


def _solve_dc(case: dict[str, Any], injections: np.ndarray | None) -> np.ndarray:
    """Solve the reduced linear system fixing the slack angle to zero."""

    buses = case["buses"]
    slack = case["slack"]
    Y = case["admittance"]

    inj = case["injections"] if injections is None else injections
    if inj.shape[0] != buses.shape[0]:
        raise ValueError("Injection vector length mismatch")

    mask = np.ones_like(buses, dtype=bool)
    mask[slack] = False

    Y_red = Y[np.ix_(mask, mask)]
    P_red = inj[mask]

    theta = np.zeros_like(buses, dtype=float)
    theta[mask] = np.linalg.solve(Y_red, P_red)

    residual = Y @ theta - inj
    if not np.allclose(residual, 0.0, atol=1e-6):
        raise RuntimeError("Power balance residual exceeds tolerance")

    return theta

