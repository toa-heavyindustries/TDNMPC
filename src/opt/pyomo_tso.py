"""Upper-layer TSO DC optimisation using Pyomo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from utils.pyomo_utils import array_to_indexed_dict
from utils.solver import finalize_solver_choice


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
    """Create a linear DC optimisation model for the TSO layer.

    Minor refactor to helper functions to reduce cyclomatic complexity.
    """

    (
        Y,
        injections,
        boundary,
        boundary_targets,
        rho,
        lower_bounds,
        upper_bounds,
        all_buses,
        boundary_list,
        internal_list,
    ) = _prepare_tso_inputs(params)

    model = pyo.ConcreteModel("TSO_DC")
    _init_sets_and_params(
        model,
        all_buses,
        boundary_list,
        internal_list,
        Y,
        injections,
        boundary,
        boundary_targets,
        lower_bounds,
        upper_bounds,
        rho,
        params.cost_coeff,
    )
    _init_variables(model, all_buses)
    _init_constraints(model)
    _init_objective(model)
    return model


def _prepare_tso_inputs(
    params: TSOParameters,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray | None,
    np.ndarray | None,
    list[int],
    list[int],
    list[int],
]:
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
    return (
        Y,
        injections,
        boundary,
        boundary_targets,
        rho,
        lower_bounds,
        upper_bounds,
        all_buses,
        boundary_list,
        internal_list,
    )


def _init_sets_and_params(
    model: pyo.ConcreteModel,
    all_buses: list[int],
    boundary_list: list[int],
    internal_list: list[int],
    Y: np.ndarray,
    injections: np.ndarray,
    boundary: np.ndarray,
    boundary_targets: np.ndarray,
    lower_bounds: np.ndarray | None,
    upper_bounds: np.ndarray | None,
    rho: float,
    cost_coeff: float,
) -> None:
    model.B = pyo.Set(initialize=all_buses)
    model.boundary = pyo.Set(within=model.B, initialize=boundary_list)
    model.internal = pyo.Set(within=model.B, initialize=internal_list)
    Y_dict = array_to_indexed_dict(Y, (all_buses, all_buses))
    model.Y = pyo.Param(model.B, model.B, initialize=Y_dict, mutable=False)
    target_map = {int(b): float(t) for b, t in zip(boundary, boundary_targets, strict=False)}
    low_def = np.full_like(boundary, -np.inf, dtype=float) if lower_bounds is None else lower_bounds
    up_def = np.full_like(boundary, np.inf, dtype=float) if upper_bounds is None else upper_bounds
    lower_map = {int(b): float(val) for b, val in zip(boundary, low_def, strict=False)}
    upper_map = {int(b): float(val) for b, val in zip(boundary, up_def, strict=False)}
    internal_map = {int(b): float(injections[b]) for b in internal_list}
    model.P_target = pyo.Param(model.boundary, initialize=target_map, mutable=False)
    model.P_internal = pyo.Param(model.internal, initialize=internal_map, mutable=False)
    model.P_lower = pyo.Param(model.boundary, initialize=lambda m, i: lower_map[int(i)], mutable=False)
    model.P_upper = pyo.Param(model.boundary, initialize=lambda m, i: upper_map[int(i)], mutable=False)
    model.cost_coeff = pyo.Param(initialize=float(cost_coeff), mutable=False)
    model.rho_param = pyo.Param(initialize=float(rho), mutable=False)


def _init_variables(model: pyo.ConcreteModel, all_buses: list[int]) -> None:
    model.theta = pyo.Var(model.B, domain=pyo.Reals, initialize=0.0)
    model.p_boundary = pyo.Var(model.boundary, domain=pyo.Reals)
    model.p_adj_pos = pyo.Var(model.internal, domain=pyo.NonNegativeReals)
    model.p_adj_neg = pyo.Var(model.internal, domain=pyo.NonNegativeReals)
    model.p_adj = pyo.Expression(model.internal, rule=lambda m, i: m.p_adj_pos[i] - m.p_adj_neg[i])
    slack_bus = min(all_buses)
    model.theta[slack_bus].fix(0.0)


def _init_constraints(model: pyo.ConcreteModel) -> None:
    def boundary_balance_rule(m, i):
        return _flow_expression(m, i) == m.p_boundary[i]

    def internal_balance_rule(m, i):
        return _flow_expression(m, i) == m.P_internal[i] + m.p_adj[i]

    def boundary_limits_rule(m, i):
        return pyo.inequality(m.P_lower[i], m.p_boundary[i], m.P_upper[i])

    model.boundary_balance = pyo.Constraint(model.boundary, rule=boundary_balance_rule)
    model.internal_balance = pyo.Constraint(model.internal, rule=internal_balance_rule)
    model.boundary_limits = pyo.Constraint(model.boundary, rule=boundary_limits_rule)


def _init_objective(model: pyo.ConcreteModel) -> None:
    def cost_adj_rule(m):
        return m.cost_coeff * sum(m.p_adj_pos[i] + m.p_adj_neg[i] for i in m.internal)

    def cost_penalty_rule(m):
        return (m.rho_param / 2.0) * sum((m.p_boundary[i] - m.P_target[i]) ** 2 for i in m.boundary)

    model.cost_adj_expr = pyo.Expression(rule=cost_adj_rule)
    model.cost_penalty_expr = pyo.Expression(rule=cost_penalty_rule)
    model.objective = pyo.Objective(expr=model.cost_adj_expr + model.cost_penalty_expr, sense=pyo.minimize)


def solve_tso_model(
    model: pyo.ConcreteModel,
    solver: str = "gurobi",
    options: dict[str, Any] | None = None,
) -> pyo.SolverResults:
    """Solve the Pyomo TSO model using the specified solver."""

    # One-solver policy: gurobi or ipopt (configured in YAML)
    selected, mapped_options = finalize_solver_choice(solver, options)
    solver_obj = pyo.SolverFactory(selected)
    if solver_obj is None or not solver_obj.available():
        raise RuntimeError(f"Solver {selected} is not available")

    def _run(name: str, opts: dict[str, Any]) -> pyo.SolverResults:
        obj = pyo.SolverFactory(name)
        if obj is None or not obj.available():
            raise RuntimeError(f"Solver {name} is not available")
        return obj.solve(model, tee=False, options=opts)

    # No GLPK path; model is LP/QP acceptable to both Gurobi and Ipopt

    results = _run(selected, mapped_options)

    termination = results.solver.termination_condition
    status = results.solver.status
    accepted = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.locallyOptimal,
        # Accept time/iteration limits as long as we have a candidate solution
        pyo.TerminationCondition.maxTimeLimit,
        pyo.TerminationCondition.maxIterations,
    }
    if (status != pyo.SolverStatus.ok) and (termination not in accepted):
        raise RuntimeError(
            "TSO optimisation failed: " f"status={status}, termination={termination}"
        )
    return results


def extract_solution(model: pyo.ConcreteModel, params: TSOParameters) -> TSOResult:
    """Extract voltage angles, flows, and adjustments as pandas structures."""

    Y = np.asarray(params.admittance, dtype=float)
    all_buses = sorted(model.B)

    theta_vals = np.array([pyo.value(model.theta[b]) for b in all_buses])
    # Match the Pyomo flow expression: sum_j Y[b,j] * (theta[b] - theta[j])
    flows = np.zeros_like(theta_vals)
    for idx, b in enumerate(all_buses):
        flows[idx] = float(np.sum(Y[b, :] * (theta_vals[idx] - theta_vals)))

    adjustments = np.zeros(len(all_buses))
    for b in model.internal:
        adjustments[b] = pyo.value(model.p_adj[b])

    theta_series = pd.Series(theta_vals, index=all_buses, name="theta")
    flow_series = pd.Series(flows, index=all_buses, name="flow")
    adj_series = pd.Series(adjustments, index=all_buses, name="p_adj")
    obj_value = float(pyo.value(model.objective))

    return TSOResult(theta=theta_series, flows=flow_series, adjustments=adj_series, objective=obj_value)
