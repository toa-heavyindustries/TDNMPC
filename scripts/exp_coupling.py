"""Assemble coupled TSO(case118) + 3x CIGRE MV feeders via transformers.

Implements step 2 of 主要实验.md: auto-select top-3 TSO load buses, merge
nets with interface transformers, and verify AC power flow of the combined system.

Usage:
  uv run python scripts/exp_coupling.py --tag coupled
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
from dso.network import build_ieee_european_lv_feeder
from utils.config import ensure_run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None, help="Run directory tag")
    p.add_argument("--case", default="case118", choices=["case39", "case118"], help="TSO case name")
    p.add_argument("--feeder-peak", type=float, default=20.0, help="Per-feeder peak MW (MV feeders)")
    p.add_argument(
        "--trafo-mva",
        type=float,
        default=25.0,
        help="Transformer MVA rating (choose 25 or 40 per 主要实验.md)",
    )
    p.add_argument(
        "--attach-lv",
        action="store_true",
        help="Attach one IEEE European LV feeder at a selected boundary bus (default last)",
    )
    p.add_argument(
        "--lv-scenario",
        default="on_mid",
        help="LV scenario key (on_mid/off_start/off_end or raw pandapower key)",
    )
    p.add_argument(
        "--lv-boundary-index",
        type=int,
        default=-1,
        help="Index into discovered boundary buses to attach LV (0-based, default -1 for last)",
    )
    return p.parse_args(argv)


def _pick_spec(mva: float) -> TransformerSpec:
    if mva <= 25.0:
        return TransformerSpec(sn_mva=25.0, vk_percent=12.0, vkr_percent=0.41, pfe_kw=14.0, i0_percent=0.07)
    return TransformerSpec(sn_mva=40.0, vk_percent=16.2, vkr_percent=0.34, pfe_kw=18.0, i0_percent=0.05)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = ensure_run_dir(args.tag)

    # Ensure boundary selection on case118 picks top-3 load buses (implicitly via plan)
    spec = _pick_spec(args.trafo_mva)

    plan: BaselineCoupledSystem = plan_baseline_coupled_system(
        args.case,
        feeder_type="mv",
        feeder_peak_mw=args.feeder_peak,
        transformer_spec=spec,
    )

    if args.attach_lv and plan.attachments:
        lv = build_ieee_european_lv_feeder(args.lv_scenario)
        idx = args.lv_boundary_index if args.lv_boundary_index != -1 else (len(plan.attachments) - 1)
        if not (0 <= idx < len(plan.attachments)):
            raise SystemExit(f"lv-boundary-index out of range [0,{len(plan.attachments)-1}]")
        plan.attachments[idx].feeder = lv

    net = assemble_baseline_network(plan)

    # AC feasibility check
    pp.runpp(net, calculate_voltage_angles=True, init="results")
    vm = net.res_bus.vm_pu.to_numpy(copy=False)
    loading_line = net.res_line.loading_percent.to_numpy(copy=False) if not net.line.empty else np.array([])
    loading_trafo = net.res_trafo.loading_percent.to_numpy(copy=False) if not net.trafo.empty else np.array([])
    metrics = {
        "vm_min": float(vm.min()) if vm.size else np.nan,
        "vm_max": float(vm.max()) if vm.size else np.nan,
        "line_max_loading": float(loading_line.max()) if loading_line.size else 0.0,
        "trafo_max_loading": float(loading_trafo.max()) if loading_trafo.size else 0.0,
        "n_bus": int(len(net.bus)),
        "n_line": int(len(net.line)),
        "n_trafo": int(len(net.trafo)),
        "boundary_names": plan.tso.boundary_bus_names,
    }

    (run_dir / "coupled.metrics.json").write_text(json.dumps(metrics, indent=2))
    pp.to_json(net, run_dir / "coupled_net.json")


if __name__ == "__main__":
    main()
