"""Baseline snapshots per 主要实验.md: TSO(case118) and DSO(CIGRE MV/LV).

Runs AC power flow, checks voltage compliance, and writes metrics to runs/<ts>/.

Usage:
  uv run python scripts/exp_baseline.py --tag baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from dso.network import build_cigre_feeder
from tso.network import build_tso_pandapower
from utils.config import ensure_run_dir


def _run_ac(net: pp.pandapowerNet) -> dict[str, Any]:
    pp.runpp(net, calculate_voltage_angles=True, init="results")
    vm = net.res_bus.vm_pu.to_numpy(copy=False)
    loading_line = net.res_line.loading_percent.to_numpy(copy=False) if not net.line.empty else np.array([])
    loading_trafo = net.res_trafo.loading_percent.to_numpy(copy=False) if not net.trafo.empty else np.array([])
    return {
        "vm_min": float(vm.min()) if vm.size else np.nan,
        "vm_max": float(vm.max()) if vm.size else np.nan,
        "line_max_loading": float(loading_line.max()) if loading_line.size else 0.0,
        "trafo_max_loading": float(loading_trafo.max()) if loading_trafo.size else 0.0,
        "n_bus": int(len(net.bus)),
        "n_line": int(len(net.line)),
        "n_trafo": int(len(net.trafo)),
    }


def _compliance(m: dict[str, Any], vmin: float = 0.95, vmax: float = 1.05) -> dict[str, Any]:
    ok_v = (np.isnan(m["vm_min"]) or m["vm_min"] >= vmin) and (np.isnan(m["vm_max"]) or m["vm_max"] <= vmax)
    ok_line = m["line_max_loading"] <= 100.0 + 1e-6
    ok_trafo = m["trafo_max_loading"] <= 100.0 + 1e-6
    return {"voltage_ok": bool(ok_v), "line_ok": bool(ok_line), "trafo_ok": bool(ok_trafo)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None, help="Run directory tag")
    p.add_argument("--include-lv", action="store_true", help="Also run IEEE European LV on_mid snapshot")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = ensure_run_dir(args.tag)

    # TSO: IEEE 118
    tso = build_tso_pandapower("case118")
    tso_metrics = _run_ac(tso.net)
    tso_metrics |= {"case": "case118"}
    tso_metrics |= _compliance(tso_metrics)
    (run_dir / "tso_case118.metrics.json").write_text(json.dumps(tso_metrics, indent=2))

    # DSO: CIGRE MV with DERs where available
    cigre_mv = build_cigre_feeder("mv", with_der=False)  # deterministic baseline
    mv_metrics = _run_ac(cigre_mv.net)
    mv_metrics |= {"case": "cigre_mv"}
    mv_metrics |= _compliance(mv_metrics)
    (run_dir / "dso_cigre_mv.metrics.json").write_text(json.dumps(mv_metrics, indent=2))

    # Optional: IEEE European LV asymmetric (on_mid)
    if args.include_lv:
        try:
            lv = pn.ieee_european_lv_asymmetric(scenario="on_mid")
            lv_metrics = _run_ac(lv)
            lv_metrics |= {"case": "ieee_european_lv_on_mid"}
            lv_metrics |= _compliance(lv_metrics)
            (run_dir / "dso_ieee_lv.metrics.json").write_text(json.dumps(lv_metrics, indent=2))
        except Exception as exc:  # pragma: no cover - optional path
            (run_dir / "dso_ieee_lv.error.txt").write_text(str(exc))

    # Summary
    summary = {
        "tso": tso_metrics,
        "dso_mv": mv_metrics,
    }
    if (run_dir / "dso_ieee_lv.metrics.json").exists():
        summary["dso_lv"] = json.loads((run_dir / "dso_ieee_lv.metrics.json").read_text())
    (run_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

