"""Solver utilities for Pyomo: single-solver selection and time limits.

This project uses exactly one solver per run, configured as either
``gurobi`` or ``ipopt``. This module validates solver choice and maps a
generic time limit to the solver-specific option.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo


def is_solver_available(name: str) -> bool:
    obj = pyo.SolverFactory(name)
    return bool(obj and obj.available())


def finalize_solver_choice(
    solver: str | None,
    options: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Validate solver choice (gurobi|ipopt) and map options.

    Returns the lowercase solver name and the mapped options dict.
    """
    name = (solver or "").lower()
    if name not in {"gurobi", "ipopt"}:
        raise ValueError("solver must be 'gurobi' or 'ipopt'")
    mapped = map_time_limit_option(name, options or {})
    return name, mapped


def map_time_limit_option(solver: str, options: dict[str, Any]) -> dict[str, Any]:
    """Map a generic time limit to solver-specific parameters.

    Supports a generic option ``time_limit_seconds``. Returns a new dict.
    """

    opts = dict(options) if options else {}
    tl = None
    # Allow a few aliases
    for key in ("time_limit_seconds", "time_limit", "timelimit", "max_time"):
        if key in opts:
            tl = opts.pop(key)
            break
    if tl is None:
        return opts

    try:
        tl_val = float(tl)
    except Exception:
        return opts

    s = solver.lower()
    if s == "gurobi":
        # Pyomo GUROBI shell plugin reads 'TimeLimit'
        opts.setdefault("TimeLimit", tl_val)
    elif s == "ipopt":
        # Ipopt uses 'max_cpu_time'
        opts.setdefault("max_cpu_time", tl_val)
    else:
        # Should not happen: validate earlier
        raise ValueError("Unsupported solver for time limit mapping")
    # Others: pass-through untouched
    return opts
