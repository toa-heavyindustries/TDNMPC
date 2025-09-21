"""Pandapower helper functions for distribution network modelling.

Adds convenience wrappers for AC solve and sensitivity extraction to support
the basic experiments pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd

from models.lindistflow import linearize_lindistflow
import pandapower.networks as pn

IEE33_LINE_DATA = [
    (1, 2, 0.0922, 0.0470),
    (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),
    (6, 7, 0.1872, 0.6188),
    (7, 8, 1.7114, 1.2351),
    (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238),
    (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129),
    (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),
    (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),
    (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 1.3400),
    (29, 30, 0.5075, 0.2585),
    (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619),
    (32, 33, 0.3410, 0.5302),
]

LOADS_KW = {
    2: (100, 60),
    3: (90, 40),
    4: (120, 80),
    5: (60, 30),
    6: (60, 35),
    7: (200, 100),
    8: (200, 100),
    9: (60, 20),
    10: (60, 20),
    11: (45, 30),
    12: (60, 35),
    13: (72, 40),
    14: (72, 40),
    15: (36, 20),
    16: (36, 20),
    17: (60, 20),
    18: (60, 20),
    19: (90, 40),
    20: (90, 40),
    21: (90, 40),
    22: (90, 40),
    23: (90, 50),
    24: (420, 200),
    25: (420, 200),
    26: (60, 25),
    27: (60, 25),
    28: (60, 20),
    29: (120, 70),
    30: (200, 600),
    31: (150, 70),
    32: (210, 100),
    33: (60, 20),
}


def build_ieee33(base_kv: float = 12.66) -> pp.pandapowerNet:
    """Construct a single-phase IEEE 33-bus distribution test feeder.

    Parameters
    ----------
    base_kv:
        Base voltage level in kV for the MV grid.

    Returns
    -------
    pp.pandapowerNet
        Pandapower network object with bus, line, load, and slack definitions.
    """

    net = pp.create_empty_network(sn_mva=100.0)

    buses = {1: pp.create_bus(net, vn_kv=base_kv, name="Bus 1")}
    for bus_idx in range(2, 34):
        buses[bus_idx] = pp.create_bus(net, vn_kv=base_kv, name=f"Bus {bus_idx}")

    pp.create_ext_grid(net, buses[1], vm_pu=1.0, name="Slack")

    for bus_idx, (p_kw, q_kw) in LOADS_KW.items():
        pp.create_load(
            net,
            bus=buses[bus_idx],
            p_mw=p_kw / 1000.0,
            q_mvar=q_kw / 1000.0,
            name=f"Load {bus_idx}",
        )

    for from_bus, to_bus, r, x in IEE33_LINE_DATA:
        pp.create_line_from_parameters(
            net,
            from_bus=buses[from_bus],
            to_bus=buses[to_bus],
            length_km=1.0,
            r_ohm_per_km=r,
            x_ohm_per_km=x,
            c_nf_per_km=0.0,
            max_i_ka=0.4,
            name=f"Line {from_bus}-{to_bus}",
        )

    return net


def ac_power_flow(net: pp.pandapowerNet) -> pd.DataFrame:
    """Run an AC power flow and return bus-level voltage and load data."""

    pp.runpp(net, algorithm="nr", numba=False, enforce_q_lims=True)

    bus = net.res_bus
    result = pd.DataFrame(index=bus.index)
    result["vm_pu"] = bus.vm_pu
    result["va_degree"] = bus.va_degree
    result["p_kw"] = net.res_load.p_mw.reindex(bus.index, fill_value=0.0) * 1000
    result["q_kw"] = net.res_load.q_mvar.reindex(bus.index, fill_value=0.0) * 1000
    return result


def export_net(net: pp.pandapowerNet, path: Path) -> None:
    """Serialise a pandapower network to disk using JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, path)


def load_net(path: Path) -> pp.pandapowerNet:
    """Load a pandapower network serialized by :func:`export_net`."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Network file not found: {path}")
    return pp.from_json(path)


def solve_ac(net: pp.pandapowerNet) -> pd.DataFrame:
    """Alias for :func:`ac_power_flow` for API consistency with the spec."""

    return ac_power_flow(net)


def get_sensitivity(
    net: pp.pandapowerNet,
    method: str = "lindistflow",
    at_state: pd.DataFrame | None = None,
    *,
    epsilon: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Return voltage sensitivity matrices S = [Rp, Rq] and base voltage.

    Parameters
    ----------
    net:
        Pandapower network (will be solved if ``at_state`` is not provided).
    method:
        "lindistflow" (default) or "local". Both currently compute numeric
        Jacobians around the operating point using small perturbations.
    at_state:
        Optional DataFrame from a prior AC solve providing the base state.
    epsilon:
        Perturbation magnitude (MW / MVAr) used when computing the Jacobian.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``Rp``, ``Rq``, and ``vm_base``.
    """

    if at_state is None:
        at_state = ac_power_flow(net)

    # linearize_lindistflow already performs a local numerical sensitivity around at_state
    sens = linearize_lindistflow(net, at_state)
    # Note: epsilon kept for future alternative implementations
    _ = epsilon

    if method not in {"lindistflow", "local"}:
        raise ValueError("Unsupported sensitivity method: " + method)
    return sens


@dataclass(slots=True)
class DsoFeeder:
    """Distribution feeder specification built from pandapower templates."""

    net: pp.pandapowerNet
    feeder_type: Literal["mv", "lv"]
    target_peak_mw: float
    cos_phi: float
    root_bus: int


def build_cigre_feeder(
    feeder_type: Literal["mv", "lv"] = "mv",
    *,
    target_peak_mw: float | None = None,
    cos_phi: float = 0.95,
    with_der: bool = False,
) -> DsoFeeder:
    """Construct a CIGRE MV/LV feeder scaled to the desired peak loading."""

    feeder_key = feeder_type.lower()
    if feeder_key not in {"mv", "lv"}:
        raise ValueError("feeder_type must be 'mv' or 'lv'")

    if target_peak_mw is None:
        target_peak_mw = 30.0 if feeder_key == "mv" else 0.4

    if feeder_key == "mv":
        net = pn.create_cigre_network_mv(with_der=with_der)
    else:
        if with_der:
            raise ValueError("with_der is not supported for CIGRE LV templates")
        net = pn.create_cigre_network_lv()

    if "min_vm_pu" in net.bus.columns:
        net.bus.loc[:, "min_vm_pu"] = 0.95
    if "max_vm_pu" in net.bus.columns:
        net.bus.loc[:, "max_vm_pu"] = 1.05

    _scale_feeder_loads(net, target_peak_mw, cos_phi)

    root_bus = int(net.ext_grid.bus.iloc[0]) if not net.ext_grid.empty else 0

    net.bus["is_boundary"] = False
    if root_bus in net.bus.index:
        net.bus.at[root_bus, "is_boundary"] = True

    feeder_literal: Literal["mv", "lv"] = "mv" if feeder_key == "mv" else "lv"

    return DsoFeeder(
        net=net,
        feeder_type=feeder_literal,
        target_peak_mw=target_peak_mw,
        cos_phi=cos_phi,
        root_bus=root_bus,
    )


def build_ieee_european_lv_feeder(
    scenario: str = "on_mid",
    *,
    cos_phi: float = 0.95,
    target_peak_mw: float | None = None,
) -> DsoFeeder:
    """Construct an IEEE European LV three-phase feeder wrapped as DsoFeeder.

    The underlying pandapower constructor expects specific scenario keys which can vary
    across versions. We accept common aliases (on_mid/off_start/off_end) and try direct
    usage otherwise.
    """

    # Map friendly names to known keys if necessary
    alias = {
        "on_mid": "on_peak_566",
        "off_start": "off_peak_1",
        "off_end": "off_peak_1440",
    }
    key = alias.get(scenario, scenario)
    try:
        net = pn.ieee_european_lv_asymmetric(scenario=key)
    except Exception:
        # Fallback: try without scenario (library default)
        net = pn.ieee_european_lv_asymmetric()

    # Harmonise voltage limits
    if "min_vm_pu" in net.bus.columns:
        net.bus.loc[:, "min_vm_pu"] = 0.95
    if "max_vm_pu" in net.bus.columns:
        net.bus.loc[:, "max_vm_pu"] = 1.05

    # Determine root PCC bus
    root_bus = int(net.ext_grid.bus.iloc[0]) if hasattr(net, "ext_grid") and not net.ext_grid.empty else int(net.bus.index[0])

    # Target peak default for LV
    if target_peak_mw is None:
        target_peak_mw = 0.4

    # Mark boundary bus
    net.bus["is_boundary"] = False
    if root_bus in net.bus.index:
        net.bus.at[root_bus, "is_boundary"] = True

    return DsoFeeder(
        net=net,
        feeder_type="lv",
        target_peak_mw=float(target_peak_mw),
        cos_phi=float(cos_phi),
        root_bus=root_bus,
    )


def _scale_feeder_loads(net: pp.pandapowerNet, target_peak_mw: float, cos_phi: float) -> None:
    """Uniformly scale loads to match ``target_peak_mw`` at the specified cosφ."""

    total_p = float(net.load.p_mw.sum())
    if not total_p:
        return

    scale = target_peak_mw / total_p
    net.load.loc[:, "p_mw"] *= scale

    cos_phi = float(cos_phi)
    cos_phi = np.clip(cos_phi, 0.1, 0.999)
    tan_phi = np.sqrt(1.0 / (cos_phi**2) - 1.0)
    net.load.loc[:, "q_mvar"] = net.load.p_mw * tan_phi
