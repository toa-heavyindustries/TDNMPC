"""Trajectory improvement (TI) envelope utilities.

Includes both lightweight smoothing primitives (Envelope dataclass) used by
controllers and higher-level helpers aligned with the README Step 9 API:
  - compute_envelope(history, bands)
  - apply_envelope_to_model(model, env, vars_map)
  - update_envelope(env, new_obs)
The higher-level helpers are thin adapters around existing building blocks
to keep responsibilities small while satisfying the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Envelope:
    """Stores smoothed lower/upper bounds for interface signals."""

    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    margin: float
    count: int = 0

    def as_dict(self) -> dict[str, list[float] | float | int]:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "alpha": self.alpha,
            "margin": self.margin,
            "count": self.count,
        }


def create_envelope(size: int, margin: float = 0.05, alpha: float = 0.2) -> Envelope:
    """Initialise an envelope with symmetric margins around zero."""

    if size <= 0:
        raise ValueError("size must be positive")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    lower = -np.ones(size) * margin
    upper = np.ones(size) * margin
    return Envelope(lower=lower, upper=upper, alpha=float(alpha), margin=float(margin))


def update_envelope(env: Envelope, values: np.ndarray) -> Envelope:
    """Update envelope bounds using exponential smoothing towards new extrema."""

    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != env.lower.shape[0]:
        raise ValueError("value dimension mismatch")

    target_lower = np.minimum(env.lower, arr - env.margin)
    target_upper = np.maximum(env.upper, arr + env.margin)

    env.lower = (1.0 - env.alpha) * env.lower + env.alpha * target_lower
    env.upper = (1.0 - env.alpha) * env.upper + env.alpha * target_upper

    env.lower = np.minimum(env.lower, arr - env.margin)
    env.upper = np.maximum(env.upper, arr + env.margin)
    env.count += 1
    return env


def violation_stats(env: Envelope, values: np.ndarray) -> dict[str, float]:
    """Compute violation metrics for a batch of values."""

    arr = np.asarray(values, dtype=float)
    if arr.shape[-1] != env.lower.shape[0]:
        raise ValueError("value dimension mismatch")

    violations_low = np.maximum(env.lower - arr, 0.0)
    violations_high = np.maximum(arr - env.upper, 0.0)
    total_violation = violations_low + violations_high

    max_violation = float(total_violation.max()) if total_violation.size else 0.0
    l2_violation = float(np.linalg.norm(total_violation.ravel()))
    rate = float(np.mean(total_violation > 0)) if total_violation.size else 0.0

    return {
        "max": max_violation,
        "l2": l2_violation,
        "rate": rate,
    }


def compute_bounds_from_scenarios(
    scenarios: np.ndarray,
    *,
    alpha: float = 1.0,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-varying lower/upper bounds from multi-scenario trajectories.

    Parameters
    ----------
    scenarios:
        Array with shape (n_scenarios, horizon, n_signals).
    alpha:
        Tightening factor in (0, 1]. 1.0 keeps the convex hull [min, max]; lower values
        shrink bounds towards the midpoint.
    margin:
        Non-negative padding added to upper and subtracted from lower.

    Returns
    -------
    (lower, upper):
        Bounds with shape (horizon, n_signals).
    """

    arr = np.asarray(scenarios, dtype=float)
    if arr.ndim != 3:
        raise ValueError("scenarios must be 3D: (n_scen, horizon, n_signals)")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    s_min = np.min(arr, axis=0)
    s_max = np.max(arr, axis=0)
    center = 0.5 * (s_min + s_max)
    halfspan = 0.5 * (s_max - s_min)
    halfspan *= alpha

    lower = center - halfspan - margin
    upper = center + halfspan + margin
    return lower, upper


# ---------------------------------------------------------------------------
# README Step 9 compatibility helpers
# ---------------------------------------------------------------------------

def compute_envelope(history: pd.DataFrame, bands: dict[str, float]) -> dict:
    """Compute TI envelope from recorded multi-step horizon predictions in history.

    Looks for columns 'dso_h' or 'tso_h' containing lists (T x size). Stacks
    them as scenarios, then computes bounds via compute_bounds_from_scenarios
    using optional keys in bands: 'alpha' (default 1.0) and 'margin' (default 0.0).
    Falls back to the latest single horizon if only one record is present.
    """

    alpha = float(bands.get("alpha", 1.0)) if isinstance(bands, dict) else 1.0
    margin = float(bands.get("margin", 0.0)) if isinstance(bands, dict) else 0.0

    series_key = "dso_h" if "dso_h" in history.columns else ("tso_h" if "tso_h" in history.columns else None)
    if series_key is None:
        # No horizon data; create a trivial 1-step, 0-centered band
        size = len(history.iloc[-1]["tso_vector"]) if ("tso_vector" in history.columns and not history.empty) else 1
        lower = -np.ones((1, size)) * margin
        upper = np.ones((1, size)) * margin
        return {"lower": lower, "upper": upper}

    scenarios: list[np.ndarray] = []
    for _, row in history.iterrows():
        arr = np.asarray(row[series_key])  # shape (T, size)
        if arr.ndim != 2:
            continue
        scenarios.append(arr)
    if not scenarios:
        arr = np.asarray(history.iloc[-1][series_key])
        if arr.ndim == 2:
            scenarios = [arr]
        else:
            size = len(history.iloc[-1]["tso_vector"]) if "tso_vector" in history.columns else 1
            lower = -np.ones((1, size)) * margin
            upper = np.ones((1, size)) * margin
            return {"lower": lower, "upper": upper}

    S = np.stack(scenarios, axis=0)  # (n_scen, T, size)
    low, up = compute_bounds_from_scenarios(S, alpha=alpha, margin=margin)
    return {"lower": low, "upper": up}


def apply_envelope_to_model(m, env: dict, vars_map: dict) -> None:
    """Apply envelope bounds to a Pyomo model.

    Recognised mappings:
      - {'pg': {'bus_indices': [...], 'penalty': float|None, 'dt_hours': float}}
        -> applies time-varying bounds to DSO pg via opt.pyomo_dso.apply_envelope_pg_bounds
      - {'p_boundary': True}
        -> applies static bounds (first time slice) to TSO boundary variable p_boundary
    """

    lower = np.asarray(env.get("lower"))
    upper = np.asarray(env.get("upper"))
    if lower.shape != upper.shape:
        raise ValueError("Envelope lower/upper shape mismatch")

    # DSO pg bounds (time-varying)
    if "pg" in vars_map:
        try:
            from opt.pyomo_dso import apply_envelope_pg_bounds as _apply_pg
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("DSO envelope application unavailable") from exc
        spec = vars_map["pg"] or {}
        buses = list(spec.get("bus_indices", []))
        penalty = spec.get("penalty", 1000.0)
        dt_hours = spec.get("dt_hours", 1.0)
        if lower.ndim != 2 or upper.ndim != 2:
            raise ValueError("DSO envelope bounds must be 2D (T, |buses|)")
        _apply_pg(m, buses, lower, upper, penalty=penalty, dt_hours=dt_hours)

    # TSO boundary bounds (use first time slice if provided)
    if "p_boundary" in vars_map:
        try:
            import pyomo.environ as pyo  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Pyomo not available for TSO envelope application") from exc
        if lower.ndim == 2:
            l0 = lower[0, :]
            u0 = upper[0, :]
        else:
            l0 = lower
            u0 = upper
        # Align lengths with model.boundary ordering
        bound_buses = sorted(list(m.boundary)) if hasattr(m, "boundary") else []
        if len(bound_buses) != len(l0):  # pragma: no cover - defensive
            # Best effort: clip/expand
            size = min(len(bound_buses), len(l0))
            l0 = np.asarray(l0)[:size]
            u0 = np.asarray(u0)[:size]
        def _rule(mdl, i):
            idx = bound_buses.index(i)
            return pyo.inequality(float(l0[idx]), mdl.p_boundary[i], float(u0[idx]))
        m.ti_boundary_limits = pyo.Constraint(m.boundary, rule=_rule)


def update_envelope(env: dict, new_obs: dict) -> dict:
    """Update dict-based envelope bounds to include new observations.

    - If new_obs contains 'matrix' (T, size), expand time-varying bounds.
    - Else if contains 'vector' (size,), update the first time slice (or flat arrays).
    Returns the modified envelope dict.
    """

    lower = np.asarray(env.get("lower"))
    upper = np.asarray(env.get("upper"))
    mat = new_obs.get("matrix")
    vec = new_obs.get("vector")
    if mat is not None:
        A = np.asarray(mat)
        if A.shape != lower.shape:
            # broadcast first axis if possible
            if A.ndim == 2 and lower.ndim == 2 and A.shape[1] == lower.shape[1]:
                A = A[: lower.shape[0], :]
            else:
                return env
        env["lower"] = np.minimum(lower, A)
        env["upper"] = np.maximum(upper, A)
        return env
    if vec is not None:
        v = np.asarray(vec)
        if lower.ndim == 2 and v.shape[-1] == lower.shape[1]:
            env["lower"][0, :] = np.minimum(lower[0, :], v)
            env["upper"][0, :] = np.maximum(upper[0, :], v)
        elif lower.ndim == 1 and v.shape[-1] == lower.shape[0]:
            env["lower"] = np.minimum(lower, v)
            env["upper"] = np.maximum(upper, v)
    return env
