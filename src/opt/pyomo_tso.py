"""Upper-layer TSO DC optimisation using Pyomo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pyomo.environ as pyo


@dataclass
class TSOParameters:
    """Parameters required to build the TSO optimisation model."""

    admittance: np.ndarray
    injections: np.ndarray
    boundary: np.ndarray
    boundary_targets: np.ndarray
    rho: float
    cost_coeff: float = 30.0
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None


@dataclass
class TSOResult:
    """Structured output extracted from the TSO model."""

    theta: pd.Series
    flows: pd.Series
    adjustments: pd.Series
    objective: float


def _flow_expression(model: pyo.ConcreteModel, bus: int) -> pyo.Expression:
    return sum(model.Y[bus, j] * (model.theta[bus] - model.theta[j]) for j in model.B)


def build_tso_model(params: TSOParameters) -> pyo.ConcreteModel:
    """Create a linear DC optimisation model for the TSO layer."""

    Y = np.asarray(params.admittance, dtype=float)
    injections = np.asarray(params.injections, dtype=float)
    boundary = np.asarray(params.boundary, dtype=int)
    boundary_targets = np.asarray(params.boundary_targets, dtype=float)
    rho = float(params.rho)
    lower_bounds = None
    upper_bounds = None
    if params.lower_bounds is not None:
        lower_bounds = np.asarray(params.lower_bounds, dtype=float)
        if lower_bounds.shape[0] != boundary_targets.shape[0]:
            raise ValueError("Lower bounds must align with boundary targets")
    if params.upper_bounds is not None:
        upper_bounds = np.asarray(params.upper_bounds, dtype=float)
        if upper_bounds.shape[0] != boundary_targets.shape[0]:
            raise ValueError("Upper bounds must align with boundary targets")

    n_bus = Y.shape[0]
    if Y.shape != (n_bus, n_bus):
        raise ValueError("Admittance matrix must be square")
    if injections.shape[0] != n_bus:
        raise ValueError("Injection vector length mismatch")
    if boundary.size != boundary_targets.size:
        raise ValueError("Boundary targets must align with boundary buses")

    all_buses = list(range(n_bus))
    boundary_list = sorted(set(int(b) for b in boundary))
    internal_list = [b for b in all_buses if b not in boundary_list]

    model = pyo.ConcreteModel("TSO_DC")
    model.B = pyo.Set(initialize=all_buses)
    model.boundary = pyo.Set(within=model.B, initialize=boundary_list)
    model.internal = pyo.Set(within=model.B, initialize=internal_list)

    Y_dict = {(i, j): float(Y[i, j]) for i in all_buses for j in all_buses}
    model.Y = pyo.Param(model.B, model.B, initialize=Y_dict, mutable=False)

    target_map = {int(b): float(t) for b, t in zip(boundary, boundary_targets)}
    lower_default = -np.inf if lower_bounds is None else lower_bounds
    upper_default = np.inf if upper_bounds is None else upper_bounds
    lower_map = {int(b): float(val) for b, val in zip(boundary, lower_default)}
    upper_map = {int(b): float(val) for b, val in zip(boundary, upper_default)}
    internal_map = {int(b): float(injections[b]) for b in internal_list}

    model.P_target = pyo.Param(model.boundary, initialize=target_map, mutable=False)
    model.P_internal = pyo.Param(model.internal, initialize=internal_map, mutable=False)
    model.P_lower = pyo.Param(model.boundary, initialize=lambda m, i: lower_map[int(i)], mutable=False)
    model.P_upper = pyo.Param(model.boundary, initialize=lambda m, i: upper_map[int(i)], mutable=False)

    model.theta = pyo.Var(model.B, domain=pyo.Reals, initialize=0.0, bounds=(-3.14, 3.14))
    model.p_boundary = pyo.Var(model.boundary, domain=pyo.Reals)
    model.p_adj_pos = pyo.Var(model.internal, domain=pyo.NonNegativeReals)
    model.p_adj_neg = pyo.Var(model.internal, domain=pyo.NonNegativeReals)
    model.p_adj = pyo.Expression(model.internal, rule=lambda m, i: m.p_adj_pos[i] - m.p_adj_neg[i])

    slack_bus = min(all_buses)
    model.theta[slack_bus].fix(0.0)

    def boundary_balance_rule(m, i):
        return _flow_expression(m, i) == m.p_boundary[i]

    def internal_balance_rule(m, i):
        return _flow_expression(m, i) == m.P_internal[i] + m.p_adj[i]

    model.boundary_balance = pyo.Constraint(model.boundary, rule=boundary_balance_rule)
    model.internal_balance = pyo.Constraint(model.internal, rule=internal_balance_rule)

    def boundary_limits_rule(m, i):
        return pyo.inequality(m.P_lower[i], m.p_boundary[i], m.P_upper[i])

    model.boundary_limits = pyo.Constraint(model.boundary, rule=boundary_limits_rule)

    def objective_rule(m):
        cost_adj = params.cost_coeff * sum(m.p_adj_pos[i] + m.p_adj_neg[i] for i in m.internal)
        cost_admm = (rho / 2.0) * sum((m.p_boundary[i] - m.P_target[i])**2 for i in m.boundary)
        return cost_adj + cost_admm

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


def solve_tso_model(
    model: pyo.ConcreteModel, solver: str = "glpk", options: dict[str, Any] | None = None
) -> pyo.SolverResults:
    """Solve the Pyomo TSO model using the specified solver."""

    solver_obj = pyo.SolverFactory(solver)
    if solver_obj is None or not solver_obj.available():
        raise RuntimeError(f"Solver {solver} is not available")

    results = solver_obj.solve(model, tee=False, options=options or {})
    if (
        results.solver.termination_condition
        == pyo.TerminationCondition.unbounded
    ):
        raise RuntimeError("TSO optimisation failed: problem unbounded")
    # Allow other non-optimal statuses like maxIterations to pass as warnings
    return results


def extract_solution(model: pyo.ConcreteModel, params: TSOParameters) -> TSOResult:
    """Extract voltage angles, flows, and adjustments as pandas structures."""

    Y = np.asarray(params.admittance, dtype=float)
    all_buses = sorted(model.B)

    theta_vals = np.array([pyo.value(model.theta[b]) for b in all_buses])
    flows = Y @ theta_vals

    adjustments = np.zeros(len(all_buses))
    for b in model.internal:
        adjustments[b] = pyo.value(model.p_adj[b])

    theta_series = pd.Series(theta_vals, index=all_buses, name="theta")
    flow_series = pd.Series(flows, index=all_buses, name="flow")
    adj_series = pd.Series(adjustments, index=all_buses, name="p_adj")
    obj_value = float(pyo.value(model.objective))

    return TSOResult(theta=theta_series, flows=flow_series, adjustments=adj_series, objective=obj_value)
