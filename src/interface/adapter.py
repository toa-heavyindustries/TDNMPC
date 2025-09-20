"""Adapters bridging TSO boundary signals with pandapower networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandapower as pp


@dataclass(frozen=True)
class BoundaryMeasurement:
    """Container for boundary power and voltage measurements."""

    p_mw: float
    q_mvar: float
    v_pu: float
    hv_bus: int
    trafo_index: int


def measure_boundary(
    net_dso: pp.pandapowerNet,
    *,
    trafo_index: int | None = None,
    run_power_flow: bool = True,
) -> tuple[float, float, float]:
    """Return (P, Q, V) at the interface transformer high-voltage side."""

    measurement = _compute_boundary(net_dso, trafo_index, run_power_flow)
    return measurement.p_mw, measurement.q_mvar, measurement.v_pu


def apply_reference(
    net_dso: pp.pandapowerNet,
    pref_mw: float,
    qref_mvar: float,
    *,
    trafo_index: int | None = None,
    tol: float = 1e-3,
    max_iters: int = 12,
) -> BoundaryMeasurement:
    """Adjust DSO load set-points to track the requested boundary (P,Q)."""

    idx = _select_trafo_index(net_dso, trafo_index)
    state = net_dso.setdefault("adapter_state", {})  # type: ignore[assignment]
    if not isinstance(state, dict):
        state = {}
        net_dso["adapter_state"] = state

    if "base_loads" not in state:
        state["base_loads"] = net_dso.load[["p_mw", "q_mvar"]].copy()
        state["factor"] = 1.0

    base_loads = state["base_loads"]
    load_indices = base_loads.index.to_list()
    if not load_indices:
        raise ValueError("apply_reference requires the DSO net to contain loads")

    def evaluate_at_factor(factor: float) -> BoundaryMeasurement:
        net_dso.load.loc[load_indices, "p_mw"] = base_loads.loc[load_indices, "p_mw"].to_numpy(dtype=float) * factor
        net_dso.load.loc[load_indices, "q_mvar"] = base_loads.loc[load_indices, "q_mvar"].to_numpy(dtype=float) * factor
        state["factor"] = factor
        return _compute_boundary(net_dso, idx, run_power_flow=True)

    measurement = evaluate_at_factor(float(state.get("factor", 1.0)))

    if abs(measurement.p_mw - pref_mw) > tol:
        measurement = _tune_active_power(
            pref_mw,
            measurement,
            evaluate_at_factor,
            state,
            tol,
            max_iters,
        )

    for _ in range(2):
        if abs(measurement.q_mvar - qref_mvar) <= tol:
            break
        delta_q = measurement.q_mvar - qref_mvar
        _distribute_delta(net_dso, load_indices, delta_p=0.0, delta_q=delta_q)
        measurement = _compute_boundary(net_dso, idx, run_power_flow=True)
        if abs(measurement.p_mw - pref_mw) > tol:
            measurement = _tune_active_power(
                pref_mw,
                measurement,
                evaluate_at_factor,
                state,
                tol,
                max_iters,
            )

    return measurement


def replace_tso_load(
    net_tso: pp.pandapowerNet,
    bus_id: int,
    p_mw: float,
    q_mvar: float,
    *,
    name: str = "dso_interface",
) -> int:
    """Replace existing load at ``bus_id`` with an aggregated controllable load."""

    bus = int(bus_id)
    state = net_tso.setdefault("tso_interfaces", {})  # type: ignore[assignment]
    if not isinstance(state, dict):
        state = {}
        net_tso["tso_interfaces"] = state

    original = state.setdefault("original", {})
    agg = state.setdefault("aggregates", {})

    if bus not in agg:
        mask = net_tso.load.bus == bus
        if bus not in original:
            original[bus] = net_tso.load.loc[mask, ["p_mw", "q_mvar"]].copy()
        net_tso.load.loc[mask, ["p_mw", "q_mvar"]] = 0.0
        load_idx = pp.create_load(net_tso, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name, controllable=True)
        agg[bus] = load_idx
    else:
        load_idx = agg[bus]
        net_tso.load.at[load_idx, "p_mw"] = p_mw
        net_tso.load.at[load_idx, "q_mvar"] = q_mvar

    return load_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_trafo_index(net: pp.pandapowerNet, trafo_index: int | None) -> int:
    if trafo_index is not None:
        if trafo_index not in net.trafo.index:
            raise ValueError(f"Transformer index {trafo_index} not found")
        return int(trafo_index)
    if net.trafo.empty:
        raise ValueError("DSO network does not contain transformers")
    return int(net.trafo.index[0])


def _compute_boundary(
    net: pp.pandapowerNet,
    trafo_index: int | None = None,
    run_power_flow: bool = True,
) -> BoundaryMeasurement:
    idx = _select_trafo_index(net, trafo_index)
    if run_power_flow:
        pp.runpp(net, algorithm="nr", numba=False)

    hv_bus = int(net.trafo.at[idx, "hv_bus"])
    res = net.res_trafo.loc[idx]
    vm = float(net.res_bus.at[hv_bus, "vm_pu"])
    return BoundaryMeasurement(
        p_mw=float(res.p_hv_mw),
        q_mvar=float(res.q_hv_mvar),
        v_pu=vm,
        hv_bus=hv_bus,
        trafo_index=idx,
    )


def _distribute_delta(
    net: pp.pandapowerNet,
    load_indices: Iterable[int],
    delta_p: float,
    delta_q: float,
) -> None:
    load_indices = list(load_indices)
    if not load_indices:
        return

    load_slice = net.load.loc[load_indices]
    p_values = load_slice["p_mw"].to_numpy(dtype=float)
    q_values = load_slice["q_mvar"].to_numpy(dtype=float)

    if abs(delta_p) > 0:
        weights_p = np.abs(p_values)
        if weights_p.sum() == 0.0:
            weights_p = np.ones_like(p_values)
        weights_p = weights_p / weights_p.sum()
        p_values = p_values - delta_p * weights_p

    if abs(delta_q) > 0:
        weights_q = np.abs(q_values)
        if weights_q.sum() == 0.0:
            weights_q = np.ones_like(q_values)
        weights_q = weights_q / weights_q.sum()
        q_values = q_values - delta_q * weights_q

    net.load.loc[load_indices, "p_mw"] = p_values
    net.load.loc[load_indices, "q_mvar"] = q_values


def _tune_active_power(
    pref_mw: float,
    measurement: BoundaryMeasurement,
    evaluate_at_factor,
    state: dict[str, Any],
    tol: float,
    max_iters: int,
) -> BoundaryMeasurement:
    """Adjust the global load scaling factor to meet the requested P target."""

    factor = float(state.get("factor", 1.0))
    p_current = measurement.p_mw

    if abs(p_current - pref_mw) <= tol:
        return measurement

    # Determine search direction and bracket
    if pref_mw > p_current:
        f_low = factor
        meas_low = measurement
        f_high = factor
        meas_high = measurement
        for _ in range(max_iters):
            f_high *= 1.2
            meas_high = evaluate_at_factor(f_high)
            if meas_high.p_mw >= pref_mw:
                break
        else:
            return meas_high
    else:
        f_high = factor
        meas_high = measurement
        f_low = factor
        meas_low = measurement
        for _ in range(max_iters):
            f_low *= 0.8
            meas_low = evaluate_at_factor(f_low)
            if meas_low.p_mw <= pref_mw:
                break
        else:
            return meas_low

    # Ensure ordering for bisection
    if meas_low.p_mw > meas_high.p_mw:
        f_low, f_high = f_high, f_low
        meas_low, meas_high = meas_high, meas_low

    if not (meas_low.p_mw <= pref_mw <= meas_high.p_mw):
        return meas_low if abs(meas_low.p_mw - pref_mw) < abs(meas_high.p_mw - pref_mw) else meas_high

    best = meas_low
    for _ in range(max_iters):
        f_mid = 0.5 * (f_low + f_high)
        meas_mid = evaluate_at_factor(f_mid)
        best = meas_mid
        if abs(meas_mid.p_mw - pref_mw) <= tol:
            break
        if meas_mid.p_mw < pref_mw:
            f_low, meas_low = f_mid, meas_mid
        else:
            f_high, meas_high = f_mid, meas_mid

    return best
