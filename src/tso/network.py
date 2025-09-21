"""Utilities for constructing and solving transmission network test cases."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandapower as pp
import pandapower.networks as pn


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


@dataclass(slots=True)
class TsoPandapowerCase:
    """Container bundling a pandapower transmission case with metadata."""

    net: pp.pandapowerNet
    boundary_buses: list[int]
    case_name: str

    @property
    def boundary_bus_names(self) -> list[str]:
        """Return the bus ``name`` entries matching :attr:`boundary_buses`."""

        bus = self.net.bus
        return [str(bus.at[idx, "name"]) for idx in self.boundary_buses]


_PANDAPOWER_CASES: dict[str, callable[[], pp.pandapowerNet]] = {
    "case39": pn.case39,
    "case118": pn.case118,
}

_DEFAULT_BOUNDARY_LABELS: dict[str, list[int]] = {
    "case39": [16, 18, 21],
}


def build_tso_pandapower(
    case_name: str = "case39",
    boundary_buses: Sequence[int | str] | None = None,
) -> TsoPandapowerCase:
    """Return a pandapower MATPOWER test case with boundary metadata.

    Parameters
    ----------
    case_name:
        Name of the pandapower case constructor (``case39`` or ``case118``).
    boundary_buses:
        Optional iterable of bus identifiers (indices or names). If ``None``,
        defaults are selected based on the case, falling back to the first load
        buses when unspecified in :data:`_DEFAULT_BOUNDARY_LABELS`.
    """

    case_key = case_name.lower()
    try:
        factory = _PANDAPOWER_CASES[case_key]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Unsupported TSO case: {case_name}") from exc

    net = factory()
    if "min_vm_pu" in net.bus.columns:
        net.bus.loc[:, "min_vm_pu"] = 0.95
    if "max_vm_pu" in net.bus.columns:
        net.bus.loc[:, "max_vm_pu"] = 1.05
    if "vm_pu" in net.gen.columns:
        net.gen.loc[:, "vm_pu"] = np.minimum(net.gen["vm_pu"].to_numpy(), 1.028)
    boundaries = _resolve_boundary_buses(net, case_key, boundary_buses)

    net.bus["is_boundary"] = False
    if boundaries:
        net.bus.loc[boundaries, "is_boundary"] = True

    return TsoPandapowerCase(net=net, boundary_buses=boundaries, case_name=case_key)


def _resolve_boundary_buses(
    net: pp.pandapowerNet,
    case_key: str,
    boundary_buses: Sequence[int | str] | None,
) -> list[int]:
    """Resolve user-specified or default boundary buses for ``net``."""

    candidates: Iterable[int | str]
    if boundary_buses is None:
        candidates = _DEFAULT_BOUNDARY_LABELS.get(case_key, [])
        if not candidates:
            candidates = _infer_boundary_defaults(net)
    else:
        candidates = boundary_buses

    resolved: list[int] = []
    for label in candidates:
        bus_idx = _map_bus_label(net, label)
        if bus_idx not in resolved:
            resolved.append(bus_idx)
    return resolved


def _infer_boundary_defaults(net: pp.pandapowerNet) -> list[int]:
    """Select up to three PQ buses as fallback boundaries."""

    load_buses = list(dict.fromkeys(net.load.bus.tolist()))
    if not load_buses:
        return net.bus.index.tolist()[: min(3, len(net.bus))]
    return load_buses[: min(3, len(load_buses))]


def _map_bus_label(net: pp.pandapowerNet, label: int | str) -> int:
    """Translate a bus ``label`` (index or name) to the internal index."""

    bus_df = net.bus
    target_name = str(label).strip()
    name_matches = bus_df.index[bus_df["name"].astype(str) == target_name]
    if len(name_matches):
        return int(name_matches[0])

    if isinstance(label, (int, np.integer)) and int(label) in bus_df.index:
        return int(label)

    raise ValueError(f"Bus label '{label}' not found in network")
