"""Aggregate KPIs from LV snapshots, timeseries (local), and coordination runs.

Scans a run directory for known artifacts and produces:
  - report.json: structured dictionary with nested sections per module
  - report.csv: flat table with key metrics for quick comparison

Usage:
  uv run python scripts/aggregate_kpis.py <run_dir>
Examples:
  uv run python scripts/aggregate_kpis.py runs/lv_snaps
  uv run python scripts/aggregate_kpis.py runs/ts_local
  uv run python scripts/aggregate_kpis.py runs/smoke_test_our
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect_lv(run_dir: Path) -> dict[str, Any] | None:
    summary = run_dir / "lv_snapshots.summary.json"
    if not summary.exists():
        return None
    data = _load_json(summary) or {}
    # Prepare a flat KPI slice
    rows = []
    for name, m in data.items():
        rows.append(
            {
                "module": "lv_snapshots",
                "scenario": name,
                "vm_min": m.get("vm_min"),
                "vm_max": m.get("vm_max"),
                "violations": m.get("voltage_violations"),
                "line_loading_max": m.get("line_max_loading"),
                "trafo_loading_max": m.get("trafo_max_loading"),
            }
        )
    return {"summary": data, "table": rows}


def collect_timeseries(run_dir: Path) -> dict[str, Any] | None:
    summary = run_dir / "timeseries.summary.json"
    if not summary.exists():
        return None
    data = _load_json(summary) or {}
    row = {
        "module": "timeseries_local",
        "steps": data.get("steps"),
        "dt_min": data.get("dt_min"),
        "vm_min": data.get("vm_min"),
        "vm_max": data.get("vm_max"),
        "violations": data.get("voltage_violations"),
        "pv_mw_max": data.get("pv_mw_max"),
        "bess_p_mw_abs_max": data.get("bess_p_mw_abs_max"),
    }
    return {"summary": data, "table": [row]}


def collect_coordination(run_dir: Path) -> dict[str, Any] | None:
    logs = run_dir / "logs.csv"
    summary = run_dir / "summary.json"
    if not summary.exists() and not logs.exists():
        return None
    sm = _load_json(summary) or {}
    df = pd.read_csv(logs) if logs.exists() else pd.DataFrame()
    row = {
        "module": "coordination",
        "steps": sm.get("steps", len(df) if not df.empty else None),
        "interfaces": sm.get("interfaces"),
        "final_residual": sm.get("final_residual"),
        "mean_residual": sm.get("mean_residual"),
    }
    return {"summary": sm, "table": [row]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Run directory to aggregate")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir

    report: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []

    lv = collect_lv(run_dir)
    if lv:
        report["lv_snapshots"] = lv["summary"]
        flat_rows.extend(lv["table"])  # type: ignore[arg-type]

    ts = collect_timeseries(run_dir)
    if ts:
        report["timeseries_local"] = ts["summary"]
        flat_rows.extend(ts["table"])  # type: ignore[arg-type]

    co = collect_coordination(run_dir)
    if co:
        report["coordination"] = co["summary"]
        flat_rows.extend(co["table"])  # type: ignore[arg-type]

    # Write outputs
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    if flat_rows:
        df = pd.DataFrame(flat_rows)
        df.to_csv(run_dir / "report.csv", index=False)


if __name__ == "__main__":
    main()

