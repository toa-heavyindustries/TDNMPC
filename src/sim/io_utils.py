"""Runner I/O helpers: logging setup and artifact writing.

Isolated from simulation logic to keep the runner orchestration simple.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import ensure_run_dir
from utils.logging_utils import setup_structured_logging


def init_run_and_logging(cfg: dict[str, Any]) -> tuple[Path, Path, logging.Logger]:
    run_cfg = cfg.get("run", {})
    run_dir = ensure_run_dir(tag=run_cfg.get("tag"), base=run_cfg.get("base", "runs"))

    log_cfg = cfg.get("logging", {})
    log_base = Path(log_cfg.get("base_dir", "logs"))
    log_dir = log_base / run_dir.name
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = setup_structured_logging(
        log_dir,
        log_file=log_cfg.get("log_file", "runner.log"),
        level=level,
    )
    return run_dir, log_dir, logger


def write_history_and_metrics(history: list[dict[str, Any]], run_dir: Path, log_dir: Path, logger: logging.Logger, metrics_file: str = "metrics.csv") -> pd.DataFrame:
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "logs.csv", index=False)
    if not df.empty:
        metrics_cols = [col for col in ["step", "residual_max", "residual_mean"] if col in df.columns]
        if metrics_cols:
            metrics_df = df[metrics_cols].copy()
            metrics_df.to_csv(log_dir / metrics_file, index=False)
            logger.info("Metrics written to %s", log_dir / metrics_file)
    return df


def write_wide_trace(df: pd.DataFrame, run_dir: Path) -> None:
    if df.empty:
        return
    try:
        tso_mat = np.vstack(df["tso_vector"].apply(np.asarray).to_list())
        dso_mat = np.vstack(df["dso_vector"].apply(np.asarray).to_list())
        env_up = np.vstack(df["envelope_upper"].apply(np.asarray).to_list())
        env_lo = np.vstack(df["envelope_lower"].apply(np.asarray).to_list())
        cols_tso = {i: f"tso_{i}" for i in range(tso_mat.shape[1])}
        cols_dso = {i: f"dso_{i}" for i in range(dso_mat.shape[1])}
        cols_up = {i: f"env_up_{i}" for i in range(env_up.shape[1])}
        cols_lo = {i: f"env_lo_{i}" for i in range(env_lo.shape[1])}
        wide = pd.DataFrame(index=df.index)
        for i, name in cols_tso.items():
            wide[name] = tso_mat[:, i]
        for i, name in cols_dso.items():
            wide[name] = dso_mat[:, i]
        for i, name in cols_up.items():
            wide[name] = env_up[:, i]
        for i, name in cols_lo.items():
            wide[name] = env_lo[:, i]
        wide["residual_max"] = df["residual_max"].values
        wide["residual_mean"] = df["residual_mean"].values
        wide["step"] = df["step"].values
        wide.to_parquet(run_dir / "trace.parquet", index=False)
    except Exception:
        pass


def write_summary_and_figs(df: pd.DataFrame, history: list[dict[str, Any]], run_dir: Path) -> None:
    if not history:
        return
    summary = {
        "final_residual": float(history[-1]["residual_max"]),
        "mean_residual": float(df["residual_mean"].mean()) if not df.empty else None,
        "steps": int(history[-1]["step"]) + 1 if "step" in history[-1] else len(history),
        "interfaces": int(len(history[-1]["tso_vector"])) if history else 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    try:
        if not df.empty and len(df["tso_vector"].iloc[0]) > 0:
            s0_tso = [row[0] for row in df["tso_vector"]]
            s0_dso = [row[0] for row in df["dso_vector"]]
            plot_df = pd.DataFrame({"tso_0": s0_tso, "dso_0": s0_dso})
            from viz.plots import plot_timeseries

            plot_timeseries(plot_df, ["tso_0", "dso_0"], figs / "pcc_timeseries.png")
    except Exception:
        pass

    try:
        md = [
            "# Run Summary",
            "",
            f"- Steps: {summary['steps']}",
            f"- Interfaces: {summary['interfaces']}",
            f"- Final residual: {summary['final_residual']:.3e}",
            f"- Mean residual: {summary['mean_residual']:.3e}" if summary["mean_residual"] is not None else "- Mean residual: n/a",
        ]
        (run_dir / "summary.md").write_text("\n".join(md))
    except Exception:
        pass

