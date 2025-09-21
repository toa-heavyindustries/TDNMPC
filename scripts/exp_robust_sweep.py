"""Sweep key parameters (trafo MVA, BESS scale, voltage bands, tap tol) and summarize KPIs.

Usage:
  uv run python scripts/exp_robust_sweep.py --tag sweep --cases 6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandapower as pp

from utils import ensure_run_dir
from sim.base_networks import plan_baseline_coupled_system, assemble_baseline_network, TransformerSpec


@dataclass
class Case:
    trafo_mva: float
    bess_scale: float
    vmin: float
    vmax: float
    tap_tol: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None, help="Run directory tag")
    p.add_argument("--cases", type=int, default=0, help="Optional limit on number of cases (0 = all)")
    return p.parse_args(argv)


def _pick_spec(mva: float) -> TransformerSpec:
    if mva <= 25.0:
        return TransformerSpec(sn_mva=25.0, vk_percent=12.0, vkr_percent=0.41, pfe_kw=14.0, i0_percent=0.07)
    return TransformerSpec(sn_mva=40.0, vk_percent=16.2, vkr_percent=0.34, pfe_kw=18.0, i0_percent=0.05)


def _metrics(net) -> dict[str, float]:
    vm = net.res_bus.vm_pu.to_numpy(copy=False) if not net.res_bus.empty else np.array([])
    loading_line = net.res_line.loading_percent.to_numpy(copy=False) if not net.line.empty else np.array([])
    loading_trafo = net.res_trafo.loading_percent.to_numpy(copy=False) if not net.trafo.empty else np.array([])
    return {
        "vm_min": float(vm.min()) if vm.size else float("nan"),
        "vm_max": float(vm.max()) if vm.size else float("nan"),
        "line_max_loading": float(loading_line.max()) if loading_line.size else 0.0,
        "trafo_max_loading": float(loading_trafo.max()) if loading_trafo.size else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = ensure_run_dir(args.tag)
    grid = list(
        product(
            [25.0, 40.0],     # trafo MVA
            [0.5, 1.0, 1.5],  # BESS scale
            [(0.95, 1.05), (0.96, 1.04)],  # voltage bands
            [0.005, 0.01],    # tap tolerance
        )
    )
    if args.cases and args.cases > 0:
        grid = grid[: args.cases]

    records: list[dict[str, Any]] = []
    for idx, (mva, bess_scale, (vmin, vmax), tap_tol) in enumerate(grid):
        spec = _pick_spec(mva)
        plan = plan_baseline_coupled_system("case118", feeder_type="mv", feeder_peak_mw=20.0, transformer_spec=spec)
        net = assemble_baseline_network(plan)
        # Apply voltage bands
        if "min_vm_pu" in net.bus.columns:
            net.bus.loc[:, "min_vm_pu"] = float(vmin)
        if "max_vm_pu" in net.bus.columns:
            net.bus.loc[:, "max_vm_pu"] = float(vmax)

        # Add BESS per feeder root with scaling via storage p/e
        try:
            for att in net["baseline_plan"]["attachments"]:
                root = int(att["feeder_root_bus"])  # type: ignore[index]
                _ = pp.create_storage(
                    net,
                    bus=root,
                    p_mw=0.0,
                    max_e_mwh=10.0 * bess_scale,
                    min_e_mwh=1.0 * bess_scale,
                    soc_percent=50.0,
                    max_p_mw=5.0 * bess_scale,
                    min_p_mw=-5.0 * bess_scale,
                    max_q_mvar=4.0 * bess_scale,
                    min_q_mvar=-4.0 * bess_scale,
                    controllable=True,
                    name=f"bess_sweep_{root}",
                )
        except Exception:
            pass

        # Tap metadata and greedy tuning
        if not net.trafo.empty:
            mask = net.trafo["name"].astype(str).str.startswith("baseline_interface_")
            tids = net.trafo.index[mask].tolist()
            if tids:
                net.trafo.loc[tids, "tap_side"] = "hv"
                net.trafo.loc[tids, "tap_min"] = -9
                net.trafo.loc[tids, "tap_max"] = 9
                net.trafo.loc[tids, "tap_step_percent"] = 1.5
                net.trafo.loc[tids, "tap_pos"] = 0
                # Greedy tap tune towards 1.0 pu with tolerance tap_tol
                for _ in range(6):
                    try:
                        pp.runpp(net, calculate_voltage_angles=True, init="results")
                    except Exception:
                        pp.runpp(net, calculate_voltage_angles=True, init="flat")
                    changed = False
                    for tid in tids:
                        lv_bus = int(net.trafo.at[tid, "lv_bus"])
                        vm = float(net.res_bus.at[lv_bus, "vm_pu"]) if not net.res_bus.empty else 1.0
                        if abs(vm - 1.0) <= tap_tol:
                            continue
                        pos = int(net.trafo.at[tid, "tap_pos"]) if not np.isnan(net.trafo.at[tid, "tap_pos"]) else 0
                        tmin = int(net.trafo.at[tid, "tap_min"]) if not np.isnan(net.trafo.at[tid, "tap_min"]) else -9
                        tmax = int(net.trafo.at[tid, "tap_max"]) if not np.isnan(net.trafo.at[tid, "tap_max"]) else 9
                        if vm < 1.0 - tap_tol and pos < tmax:
                            net.trafo.at[tid, "tap_pos"] = pos + 1
                            changed = True
                        elif vm > 1.0 + tap_tol and pos > tmin:
                            net.trafo.at[tid, "tap_pos"] = pos - 1
                            changed = True
                    if not changed:
                        break

        # Final PF and KPI collection
        try:
            pp.runpp(net, calculate_voltage_angles=True, init="results")
        except Exception:
            pp.runpp(net, calculate_voltage_angles=True, init="flat")
        kpi = _metrics(net)
        records.append(
            {
                "case": idx,
                "trafo_mva": mva,
                "bess_scale": bess_scale,
                "vmin": vmin,
                "vmax": vmax,
                "tap_tol": tap_tol,
                **kpi,
            }
        )

    out = Path(run_dir) / "robust_sweep.json"
    out.write_text(json.dumps(records, indent=2))
    print(json.dumps({"count": len(records), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()

