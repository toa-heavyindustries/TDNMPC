"""24h timeseries (no coordination) using DFData/ConstControl.

Implements item 5 in 主要实验.md. We simulate a single CIGRE MV feeder with
local devices and controllers only (no TSO coupling):
  - Load profile drives all loads via ``scaling``
  - One PV sgen at the feeder root follows the PV profile
  - One BESS at the root follows a naive rule-based schedule derived from
    load/PV (pre-computed series)

We run ``pandapower.timeseries.run_timeseries`` with ``OutputWriter`` to log
voltages, line/trafo loading, PCC power (ext_grid), and BESS p_mw. A simple
SoC trajectory is computed ex-post from the BESS profile and written alongside
other outputs.

Usage:
  uv run python scripts/exp_timeseries_local.py --tag ts_local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dso.network import build_cigre_feeder
from utils.config import ensure_run_dir


def _import_timeseries_modules():  # lazy import with fallbacks across versions
    import importlib

    # DFData
    try:
        DFData = importlib.import_module("pandapower.timeseries").DFData  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - version-specific
        DFData = importlib.import_module("pandapower.timeseries.data_sources").DFData

    # ConstControl
    try:
        ConstControl = importlib.import_module("pandapower.control").ConstControl  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - version-specific
        ConstControl = importlib.import_module("pandapower.control.basic_controller").ConstControl

    # OutputWriter
    try:
        OutputWriter = importlib.import_module("pandapower.timeseries").OutputWriter  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - version-specific
        OutputWriter = importlib.import_module("pandapower.timeseries.output_writer").OutputWriter

    # run_timeseries
    try:
        run_timeseries = importlib.import_module("pandapower.timeseries").run_timeseries  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - version-specific
        run_timeseries = importlib.import_module("pandapower.timeseries.run" if hasattr(importlib.import_module("pandapower.timeseries"), "run" ) else "pandapower.timeseries.timeseries").run_timeseries  # type: ignore

    return DFData, ConstControl, OutputWriter, run_timeseries


def _ensure_profiles(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["time"])  # type: ignore[arg-type]
        return df
    # generate via src.profiles
    from profiles import make_profiles

    meta = make_profiles(out_dir=path.parent)
    df = pd.read_csv(meta["combined_csv"], parse_dates=["time"])  # type: ignore[arg-type]
    return df


def _build_local_dso(with_bess: bool = True, with_cap: bool = True) -> Any:
    feeder = build_cigre_feeder("mv", with_der=False)  # deterministic base
    net = feeder.net
    root = feeder.root_bus

    # Add a PV sgen at root (name: pv_root) and a BESS (name: bess_root)
    import pandapower as pp

    pv_idx = int(pp.create_sgen(net, bus=root, p_mw=0.0, q_mvar=0.0, name="pv_root", controllable=True))

    bess_idx = None
    if with_bess:
        bess_idx = int(
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
                name="bess_root",
            )
        )

    if with_cap:
        # Fixed 2 MVAr capacitor modeled as sgen injecting Q
        pp.create_sgen(net, bus=root, p_mw=0.0, q_mvar=2.0, name="cap_root", controllable=False)

    return net, root, pv_idx, bess_idx


def _rule_bess(load_kw: pd.Series, pv_kw: pd.Series, max_p_mw: float = 5.0) -> pd.Series:
    """Naive rule: charge when pv exceeds 0.6*load, else discharge lightly."""

    target_mw = (pv_kw - 0.6 * load_kw) / 1000.0
    target_mw = target_mw.clip(lower=-max_p_mw, upper=max_p_mw)
    return target_mw.rename("bess_p")


def _integrate_soc(p_mw: pd.Series, dt_min: float, e_max_mwh: float = 10.0, eta_rt: float = 0.92) -> pd.Series:
    eta = float(np.sqrt(eta_rt))
    soc = [0.5 * e_max_mwh]
    dt_h = dt_min / 60.0
    for p in p_mw.iloc[:-1]:
        if p >= 0:  # discharge
            soc.append(soc[-1] - p * dt_h / eta)
        else:  # charge
            soc.append(soc[-1] - p * dt_h * eta)
    soc_series = pd.Series(soc, index=p_mw.index, name="soc_mwh")
    soc_series = soc_series.clip(lower=0.1 * e_max_mwh, upper=0.9 * e_max_mwh)
    return soc_series


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None, help="Run directory tag")
    p.add_argument("--dt-min", type=float, default=5.0, help="Time step in minutes")
    p.add_argument("--pv-max", type=float, default=10.0, help="PV plant size in MW")
    p.add_argument("--profiles", type=Path, default=Path("data/profiles.csv"), help="CSV with time,load,pv")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = ensure_run_dir(args.tag)

    df_prof = _ensure_profiles(args.profiles)
    # Align to requested dt
    df_prof = df_prof.set_index("time").resample(f"{args.dt_min}min").interpolate(method="time").reset_index()
    n = len(df_prof)

    net, root, pv_idx, bess_idx = _build_local_dso()

    # Build timeseries DataFrame for DFData
    load_factor = (df_prof["load"] / float(df_prof["load"].max())).rename("load_factor")
    pv_p = (df_prof["pv"] / float(df_prof["pv"].max()) * args.pv_max).rename("pv_p")
    bess_p = _rule_bess(df_prof["load"], df_prof["pv"]).rename("bess_p")
    ts_df = pd.DataFrame({"time": df_prof["time"], "load_factor": load_factor, "pv_p": pv_p, "bess_p": bess_p})

    # Controllers
    DFData, ConstControl, OutputWriter, run_timeseries = _import_timeseries_modules()
    ds = DFData(ts_df)

    # Load scaling on all loads
    load_idx = net.load.index.tolist()
    if load_idx:
        _ = ConstControl(net, element="load", variable="scaling", element_index=load_idx, data_source=ds, profile_name="load_factor")

    # PV sgen p_mw
    _ = ConstControl(net, element="sgen", variable="p_mw", element_index=[pv_idx], data_source=ds, profile_name="pv_p")

    # BESS p_mw if present
    if bess_idx is not None:
        _ = ConstControl(net, element="storage", variable="p_mw", element_index=[bess_idx], data_source=ds, profile_name="bess_p")

    # Output writer
    ow = OutputWriter(net, time_steps=range(n), output_path=str(run_dir))
    # Standard logs: voltages, loading, PCC (ext_grid), BESS p
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_trafo", "loading_percent")
    if not net.ext_grid.empty:
        ow.log_variable("res_ext_grid", "p_mw")
        ow.log_variable("res_ext_grid", "q_mvar")
    if bess_idx is not None:
        ow.log_variable("res_storage", "p_mw")

    # Run timeseries
    run_timeseries(net, time_steps=range(n), output_writer=ow)

    # Compute simple SoC and save
    soc = _integrate_soc(bess_p, dt_min=float(args.dt_min))
    soc.to_csv(run_dir / "soc.csv", index=False)

    # KPI summary
    try:
        res_bus = pd.read_pickle(run_dir / "res_bus.pkl")  # wide MultiIndex columns
        vm_all = res_bus.values.flatten()
        vm_all = vm_all[np.isfinite(vm_all)]
        vmin = float(np.min(vm_all)) if vm_all.size else float("nan")
        vmax = float(np.max(vm_all)) if vm_all.size else float("nan")
        vviol = int(np.sum((vm_all < 0.95) | (vm_all > 1.05))) if vm_all.size else 0
    except Exception:
        vmin = float("nan")
        vmax = float("nan")
        vviol = 0

    summary = {
        "steps": int(n),
        "dt_min": float(args.dt_min),
        "vm_min": vmin,
        "vm_max": vmax,
        "voltage_violations": vviol,
        "pv_mw_max": float(ts_df["pv_p"].max()),
        "bess_p_mw_abs_max": float(ts_df["bess_p"].abs().max()),
    }
    (run_dir / "timeseries.summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
