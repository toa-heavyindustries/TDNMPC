"""Integration tests for the scenario runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sim.runner import simulate_scenario


def test_simulate_scenario_produces_logs(tmp_path: Path) -> None:
    cfg = {
        "time": {"start": "2024-01-01 00:00", "steps": 3, "dt_min": 15},
        "admm": {"rho": 1.0, "max_iters": 100, "tol_primal": 1e-6, "tol_dual": 1e-6},
        "signals": {
            "tso_targets": [[1.0, -0.5]] * 3,
            "dso_targets": [[0.6, -0.2]] * 3,
        },
        "envelope": {"margin": 0.05, "alpha": 0.4},
        "run": {"base": str(tmp_path), "tag": "demo"},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    result = simulate_scenario(cfg_path)
    run_dir = Path(result["run_dir"])
    assert run_dir.exists()

    logs_path = run_dir / "logs.csv"
    assert logs_path.exists()
    df = pd.read_csv(logs_path)
    assert len(df) == cfg["time"]["steps"]

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["steps"] == cfg["time"]["steps"]
    assert summary["final_residual"] < 1e-5

