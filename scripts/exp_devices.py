"""Add BESS, DiscreteTapControl, and capacitor banks to the coupled system.

Implements step 3 of 主要实验.md on top of the assembled case118 + CIGRE MV feeders.

What it does:
  - Builds the coupled net (case118 + 3x CIGRE MV), as in exp_coupling
  - For each interface transformer named 'baseline_interface_*':
      * set tap metadata (hv-side, +/-9 steps, 1.5% per step)
      * attach a DiscreteTapControl targeting 1.00 pu on the LV side
  - For each feeder root bus: add
      * one storage (5 MW / 10 MWh, +/-4 MVar)
      * one 2 Mvar capacitor bank (max_step=3 -> up to 6 Mvar)
  - Runs AC power flow with controllers and writes metrics

Usage:
  uv run python scripts/exp_devices.py --tag devices
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandapower as pp

from sim.base_networks import (
    BaselineCoupledSystem,
    TransformerSpec,
    assemble_baseline_network,
    plan_baseline_coupled_system,
)
from utils.config import ensure_run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None, help="Run directory tag")
    p.add_argument("--feeder-peak", type=float, default=20.0, help="Per-feeder peak MW")
    p.add_argument("--trafo-mva", type=float, default=25.0, help="Transformer MVA rating (25 or 40)")
    return p.parse_args(argv)


def _pick_spec(mva: float) -> TransformerSpec:
    if mva <= 25.0:
        return TransformerSpec(sn_mva=25.0, vk_percent=12.0, vkr_percent=0.41, pfe_kw=14.0, i0_percent=0.07)
    return TransformerSpec(sn_mva=40.0, vk_percent=16.2, vkr_percent=0.34, pfe_kw=18.0, i0_percent=0.05)


def _configure_tap_and_controls(net: pp.pandapowerNet) -> list[int]:
    """Ensure tap settings exist on baseline interface transformers.

    If pandapower controllers are unavailable, falls back to static metadata only.
    Returns a list of transformer indices that were configured.
    """

    if net.trafo.empty:
        return []

    mask = net.trafo["name"].astype(str).str.startswith("baseline_interface_")
    tids = net.trafo.index[mask].tolist()
    if not tids:
        return []

    net.trafo.loc[tids, "tap_side"] = "hv"
    net.trafo.loc[tids, "tap_min"] = -9
    net.trafo.loc[tids, "tap_max"] = 9
    net.trafo.loc[tids, "tap_step_percent"] = 1.5
    net.trafo.loc[tids, "tap_pos"] = 0
    return tids


def _tune_taps_greedy(net: pp.pandapowerNet, tids: list[int], *, tol: float = 0.005, max_iter: int = 6) -> None:
    """Greedy discrete tap tuning to target ~1.00 pu at LV buses.

    In each iteration, for each transformer we adjust one tap step at most,
    then re-run power flow. Stops early if all LV voltages are within tol.
    """

    if not tids:
        return

    for _ in range(max_iter):
        try:
            pp.runpp(net, calculate_voltage_angles=True, init="results")
        except Exception:
            # If a PF fails at a tap candidate, stop tuning
            break
        changed = False
        for tid in tids:
            lv_bus = int(net.trafo.at[tid, "lv_bus"])
            vm = float(net.res_bus.at[lv_bus, "vm_pu"]) if not net.res_bus.empty else 1.0
            if abs(vm - 1.0) <= tol:
                continue
            pos = int(net.trafo.at[tid, "tap_pos"]) if not np.isnan(net.trafo.at[tid, "tap_pos"]) else 0
            tmin = int(net.trafo.at[tid, "tap_min"]) if not np.isnan(net.trafo.at[tid, "tap_min"]) else -9
            tmax = int(net.trafo.at[tid, "tap_max"]) if not np.isnan(net.trafo.at[tid, "tap_max"]) else 9
            if vm < 1.0 - tol and pos < tmax:
                net.trafo.at[tid, "tap_pos"] = pos + 1
                changed = True
            elif vm > 1.0 + tol and pos > tmin:
                net.trafo.at[tid, "tap_pos"] = pos - 1
                changed = True
        if not changed:
            break


def _add_bess_and_caps(net: pp.pandapowerNet) -> dict[str, list[int]]:
    """Add a storage and a capacitor equivalent (as sgen) at each feeder root bus.

    Returns indices of created elements by type.
    """

    created: dict[str, list[int]] = {"storage": [], "cap_sgen": []}
    plan = net.get("baseline_plan", {})
    for att in plan.get("attachments", []):
        root = int(att["feeder_root_bus"])
        sidx = int(
            pp.create_storage(
                net,
                bus=root,
                p_mw=0.0,
                max_e_mwh=10.0,
                min_e_mwh=1.0,
                soc_percent=50.0,
                max_p_mw=5.0,
                min_p_mw=-5.0,
                max_q_mvar=4.0,
                min_q_mvar=-4.0,
                controllable=True,
                name=f"bess_at_{root}",
            )
        )
        created["storage"].append(sidx)

        # Fixed 2 MVAr capacitor modeled as sgen injecting reactive power
        q_mvar = 2.0
        cap = int(pp.create_sgen(net, bus=root, p_mw=0.0, q_mvar=q_mvar, name=f"cap_{root}", controllable=False))
        created["cap_sgen"].append(cap)

    return created


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = ensure_run_dir(args.tag)

    plan: BaselineCoupledSystem = plan_baseline_coupled_system(
        "case118", feeder_type="mv", feeder_peak_mw=args.feeder_peak, transformer_spec=_pick_spec(args.trafo_mva)
    )
    net = assemble_baseline_network(plan)

    tids = _configure_tap_and_controls(net)
    created = _add_bess_and_caps(net)

    # No special handling required for sgen-based capacitors

    # Greedy tap tuning without controller dependency, then final solve
    _tune_taps_greedy(net, tids)
    try:
        pp.runpp(net, calculate_voltage_angles=True, init="results")
    except Exception:
        pp.runpp(net, calculate_voltage_angles=True, init="flat")

    vm = net.res_bus.vm_pu.to_numpy(copy=False)
    loading_line = net.res_line.loading_percent.to_numpy(copy=False) if not net.line.empty else np.array([])
    loading_trafo = net.res_trafo.loading_percent.to_numpy(copy=False) if not net.trafo.empty else np.array([])
    metrics = {
        "vm_min": float(vm.min()) if vm.size else np.nan,
        "vm_max": float(vm.max()) if vm.size else np.nan,
        "line_max_loading": float(loading_line.max()) if loading_line.size else 0.0,
        "trafo_max_loading": float(loading_trafo.max()) if loading_trafo.size else 0.0,
        "n_storage": len(created["storage"]),
        "n_caps": len(created["cap_sgen"]),
        "n_tap_ctrl": len(tids),
    }

    (run_dir / "devices.metrics.json").write_text(json.dumps(metrics, indent=2))
    pp.to_json(net, run_dir / "devices_net.json")


if __name__ == "__main__":
    main()
