"""Coupling utilities for exchanging signals between TSO and DSO layers."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from tso.network import TsoPandapowerCase


def define_coupling(
    tso_case: dict[str, Any] | TsoPandapowerCase,
    dso_nets: list[Any],
    mapping: dict[int, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Validate and assemble coupling metadata between TSO boundary buses and DSOs.

    Parameters
    ----------
    tso_case:
        Dictionary describing the DC transmission case, expected to contain a ``"boundary"``
        array of bus identifiers.
    dso_nets:
        Collection of DSO network objects (e.g. pandapower nets).
    mapping:
        Mapping from TSO boundary bus id to a tuple ``(dso_index, dso_bus_id)``.

    Returns
    -------
    dict[str, np.ndarray]
        Coupling metadata containing aligned arrays for boundary buses and their DSO targets.
    """

    boundary = _extract_boundary_array(tso_case)
    if boundary.size == 0:
        raise ValueError("TSO case does not define any boundary buses")

    tso_buses: list[int] = []
    dso_ids: list[int] = []
    dso_buses: list[int] = []

    for bus in boundary:
        if bus not in mapping:
            raise ValueError(f"Boundary bus {bus} missing from mapping")
        dso_idx, dso_bus = mapping[bus]
        if not 0 <= dso_idx < len(dso_nets):
            raise ValueError(f"DSO index {dso_idx} out of range")
        net = dso_nets[dso_idx]
        if hasattr(net, "bus") and hasattr(net.bus, "index"):
            if dso_bus not in net.bus.index:
                raise ValueError(f"DSO bus {dso_bus} not present in net {dso_idx}")
        tso_buses.append(int(bus))
        dso_ids.append(int(dso_idx))
        dso_buses.append(int(dso_bus))

    coupler = {
        "tso_buses": np.asarray(tso_buses, dtype=int),
        "dso_indices": np.asarray(dso_ids, dtype=int),
        "dso_buses": np.asarray(dso_buses, dtype=int),
        "n_interfaces": int(len(tso_buses)),
    }

    slot_groups: dict[int, list[int]] = defaultdict(list)
    for slot, dso_idx in enumerate(coupler["dso_indices"]):
        slot_groups[dso_idx].append(slot)

    for dso_idx, slots in slot_groups.items():
        net = dso_nets[dso_idx]
        coupling_info = {
            "slots": np.asarray(slots, dtype=int),
            "p_tso": np.zeros(len(slots), dtype=float),
            "p_response": np.zeros(len(slots), dtype=float),
            "dso_buses": coupler["dso_buses"][slots],
        }
        net["coupling"] = coupling_info  # type: ignore[index]

    return coupler


def _extract_boundary_array(tso_case: dict[str, Any] | TsoPandapowerCase) -> np.ndarray:
    """Return a boundary bus array for legacy dicts or new dataclass inputs."""

    if isinstance(tso_case, dict):
        return np.asarray(tso_case.get("boundary", []), dtype=int)

    boundary = getattr(tso_case, "boundary_buses", None)
    if boundary is None:
        raise ValueError("TSO case lacks boundary metadata")
    return np.asarray(boundary, dtype=int)


def push_tso_signals_to_dsos(signals: np.ndarray, dso_nets: list[Any]) -> None:
    """Scatter TSO-side power signals to each DSO according to stored coupling slots."""

    arr = np.asarray(signals, dtype=float)

    for net in dso_nets:
        coupling = net.get("coupling") if isinstance(net, dict) else getattr(net, "coupling", None)
        if not coupling:
            continue
        slots = coupling["slots"]
        if np.any(slots >= arr.shape[0]):
            raise ValueError("Signal vector shorter than coupling slots")
        coupling["p_tso"] = arr[slots]


def aggregate_dsos_to_tso(dsos: list[Any]) -> np.ndarray:
    """Collect DSO responses (``p_response``) and return them in TSO boundary order."""

    slot_records: list[tuple[np.ndarray, np.ndarray]] = []
    max_slot = -1
    for net in dsos:
        coupling = net.get("coupling") if isinstance(net, dict) else getattr(net, "coupling", None)
        if not coupling:
            continue
        slots = np.asarray(coupling["slots"], dtype=int)
        response = np.asarray(coupling.get("p_response", np.zeros(len(slots))), dtype=float)
        if response.shape[0] != slots.shape[0]:
            raise ValueError("DSO response length mismatch")
        slot_records.append((slots, response))
        if slots.size:
            max_slot = max(max_slot, int(slots.max()))

    if max_slot < 0:
        return np.zeros(0, dtype=float)

    aggregated = np.zeros(max_slot + 1, dtype=float)
    for slots, response in slot_records:
        aggregated[slots] = response
    return aggregated


def coupling_residuals(tso_p: np.ndarray, dso_p: np.ndarray) -> dict[str, float]:
    """Compute residual statistics between TSO injections and aggregated DSO responses."""

    tso_arr = np.asarray(tso_p, dtype=float)
    dso_arr = np.asarray(dso_p, dtype=float)
    if tso_arr.shape != dso_arr.shape:
        raise ValueError("TSO and DSO vectors must share the same shape")

    diff = tso_arr - dso_arr
    return {
        "max": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "l2": float(np.linalg.norm(diff)) if diff.size else 0.0,
    }
