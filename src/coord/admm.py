"""ADMM coordination utilities for aligning TSO and DSO boundary variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ADMMConfig:
    """Configuration for the consensus ADMM routine."""

    size: int
    rho: float = 1.0
    max_iters: int = 50
    tol_primal: float = 1e-3
    tol_dual: float = 1e-3


def run_admm(
    tso_update: Callable[[np.ndarray], np.ndarray],
    dso_update: Callable[[np.ndarray], np.ndarray],
    config: ADMMConfig,
    objective: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, object]:
    """Run a scaled-form consensus ADMM loop.

    Parameters
    ----------
    tso_update:
        Callable returning the updated TSO boundary vector ``x`` given the quantity ``z - u``.
    dso_update:
        Callable returning the updated DSO boundary vector ``z`` given the quantity ``x + u``.
    config:
        ADMM configuration (problem size, penalty parameter, stopping tolerances).
    objective:
        Optional callable to evaluate an objective value each iteration.

    Returns
    -------
    dict[str, object]
        Dictionary containing the final iterates (``x``, ``z``, ``u``), convergence flag,
        iteration count, and per-iteration history with residuals / objective if requested.
    """

    size = int(config.size)
    if size <= 0:
        raise ValueError("config.size must be positive")

    rho = float(config.rho)
    if rho <= 0:
        raise ValueError("config.rho must be positive")

    x = np.zeros(size, dtype=float)
    z = np.zeros(size, dtype=float)
    u = np.zeros(size, dtype=float)

    history: list[dict[str, float]] = []
    converged = False

    for iteration in range(1, config.max_iters + 1):
        x = np.asarray(tso_update(z - u), dtype=float)
        if x.shape[0] != size:
            raise ValueError("tso_update returned vector with incorrect size")

        z_prev = z.copy()
        z = np.asarray(dso_update(x + u), dtype=float)
        if z.shape[0] != size:
            raise ValueError("dso_update returned vector with incorrect size")

        u = u + x - z

        r_norm = float(np.linalg.norm(x - z))
        s_norm = float(rho * np.linalg.norm(z - z_prev))
        obj_val = float(objective(x, z)) if objective is not None else float("nan")

        history.append(
            {
                "iter": iteration,
                "primal_residual": r_norm,
                "dual_residual": s_norm,
                "objective": obj_val,
            }
        )

        if (r_norm <= config.tol_primal) and (s_norm <= config.tol_dual):
            converged = True
            break

    return {
        "x": x,
        "z": z,
        "u": u,
        "history": history,
        "converged": converged,
        "iterations": len(history),
    }



@dataclass(slots=True)
class BatchResult:
    run_dir: Path
    seeds: list[int]
    records: list[dict[str, Any]]


def make_multi_dso(n: int, base_targets: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [base_targets + rng.normal(scale=0.05, size=base_targets.shape) for _ in range(n)]


def run_batch(
    simulator: Callable[[int], dict[str, Any]], seeds: list[int], run_dir: Path
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = simulator(seed)
        record = {"seed": seed, **result}
        records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(run_dir / "batch.csv", index=False)
    return df
