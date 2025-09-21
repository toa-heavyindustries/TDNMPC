"""Pyomo-based DSO economic dispatch under LinDistFlow sensitivities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from utils.pyomo_utils import array_to_indexed_dict
from utils.timewin import Horizon


@dataclass
class DSOResult:
    """Extracted DSO optimisation results."""

    p_injections: pd.DataFrame
    voltage: pd.DataFrame
    objective: float


@dataclass
class DSOParameters:
    """Container for DSO optimisation inputs."""

    sens: dict[str, np.ndarray]
    profiles: pd.DataFrame
    horizon: Horizon
    vmin: float = 0.95
    vmax: float = 1.05
    penalty_voltage: float = 1e4
    cost_coeff: float = 50.0


def _prepare_profile_matrix(values: pd.Series, n_bus: int, steps: int) -> np.ndarray:
    arr = values.to_numpy(copy=True)
    if arr.size < steps:
        raise ValueError("Profile length shorter than horizon steps")
    base = arr[:steps]
    return np.tile(base, (n_bus, 1))


def build_dso_model(params: DSOParameters) -> pyo.ConcreteModel:
    """Construct a linear Pyomo model leveraging LinDistFlow sensitivities."""

    for column in ("load", "pv"):
        if column not in params.profiles:
            raise ValueError(f"profiles must contain column '{column}'")

    sens = params.sens
    Rp = sens["Rp"]
    Rq = sens.get("Rq", np.zeros_like(Rp))
    vm_base = sens["vm_base"]

    n_bus = Rp.shape[0]
    if Rp.shape != (n_bus, n_bus) or Rq.shape != (n_bus, n_bus):
        raise ValueError("Sensitivity matrices must be square of size n_bus")

    steps = params.horizon.steps

    load_matrix = _prepare_profile_matrix(params.profiles["load"], n_bus, steps) / 1000.0
    pv_matrix = _prepare_profile_matrix(params.profiles["pv"], n_bus, steps) / 1000.0

    model = pyo.ConcreteModel("DSO_Dispatch")
    model.B = pyo.Set(initialize=range(n_bus))
    model.T = pyo.Set(initialize=range(steps))

    buses = tuple(range(n_bus))
    times = tuple(range(steps))
    load_dict = array_to_indexed_dict(load_matrix, (buses, times))
    pv_dict = array_to_indexed_dict(pv_matrix, (buses, times))
    vm_base_dict = array_to_indexed_dict(vm_base, (buses,))
    Rp_dict = array_to_indexed_dict(Rp, (buses, buses))
    Rq_dict = array_to_indexed_dict(Rq, (buses, buses))

    model.p_load = pyo.Param(model.B, model.T, initialize=load_dict, mutable=False)
    model.p_pv = pyo.Param(model.B, model.T, initialize=pv_dict, mutable=False)
    model.vm_base = pyo.Param(model.B, initialize=vm_base_dict, mutable=False)
    model.Rp = pyo.Param(model.B, model.B, initialize=Rp_dict, mutable=False)
    model.Rq = pyo.Param(model.B, model.B, initialize=Rq_dict, mutable=False)

    model.pg = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.qg = pyo.Var(model.B, model.T, domain=pyo.Reals)
    model.v = pyo.Var(model.B, model.T)
    model.v_pos = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.v_neg = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)

    def voltage_balance_rule(m, b, t):
        active = sum(
            m.Rp[b, j] * (m.pg[j, t] - m.p_load[j, t] + m.p_pv[j, t]) for j in m.B
        )
        reactive = sum(m.Rq[b, j] * m.qg[j, t] for j in m.B)
        return m.v[b, t] == m.vm_base[b] + active + reactive

    model.voltage_balance = pyo.Constraint(model.B, model.T, rule=voltage_balance_rule)

    def voltage_upper_rule(m, b, t):
        return m.v[b, t] <= params.vmax + m.v_pos[b, t]

    def voltage_lower_rule(m, b, t):
        return m.v[b, t] >= params.vmin - m.v_neg[b, t]

    model.voltage_upper = pyo.Constraint(model.B, model.T, rule=voltage_upper_rule)
    model.voltage_lower = pyo.Constraint(model.B, model.T, rule=voltage_lower_rule)

    def zero_reactive_rule(m, b, t):
        return m.qg[b, t] == 0.0

    model.qg_zero = pyo.Constraint(model.B, model.T, rule=zero_reactive_rule)

    dt_hours = params.horizon.dt_min / 60.0

    def objective_rule(m):
        energy_cost = sum(params.cost_coeff * dt_hours * m.pg[b, t] for b in m.B for t in m.T)
        penalty = sum(
            params.penalty_voltage * (m.v_pos[b, t] + m.v_neg[b, t]) for b in m.B for t in m.T
        )
        return energy_cost + penalty

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


def solve_dso_model(model: pyo.ConcreteModel, solver: str = "glpk") -> pyo.SolverResults:
    """Solve the Pyomo model with the requested solver."""

    solver_obj = pyo.SolverFactory(solver)
    if solver_obj is None or not solver_obj.available():
        raise RuntimeError(f"Solver {solver} is not available")
    results = solver_obj.solve(model, tee=False)
    if (results.solver.status != pyo.SolverStatus.ok) or (
        results.solver.termination_condition not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}
    ):
        raise RuntimeError("DSO optimisation failed: " + str(results.solver))
    return results


def extract_solution(model: pyo.ConcreteModel, params: DSOParameters) -> DSOResult:
    """Convert Pyomo decision variables to pandas structures."""

    buses = sorted(model.B)
    times = sorted(model.T)

    pg_data = np.array([[pyo.value(model.pg[b, t]) for b in buses] for t in times])
    voltage_data = np.array([[pyo.value(model.v[b, t]) for b in buses] for t in times])

    pg_df = pd.DataFrame(pg_data, index=times, columns=buses)
    voltage_df = pd.DataFrame(voltage_data, index=times, columns=buses)
    obj = float(pyo.value(model.objective))

    return DSOResult(p_injections=pg_df, voltage=voltage_df, objective=obj)


def apply_envelope_pg_bounds(
    model: pyo.ConcreteModel,
    bus_indices: list[int],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    penalty: float | None = 1000.0,
    dt_hours: float = 1.0,
) -> None:
    """Apply time-varying bounds to ``pg`` at selected buses with optional soft penalty.

    Parameters
    ----------
    model:
        Built DSO Pyomo model (with sets ``B`` and ``T`` and variable ``pg``).
    bus_indices:
        List of bus indices (subset of ``model.B``) to which the bounds apply.
    lower, upper:
        Arrays with shape (|T|, |bus_indices|) defining bounds per time and bus.
    penalty:
        If provided (>=0), introduce non-negative slacks and add ``penalty * dt_hours``
        times the L1 slack to the objective. If ``None``, enforce hard bounds.
    dt_hours:
        Time step in hours (scales penalty to energy-equivalent if desired).
    """


    buses = list(bus_indices)
    T = sorted(model.T)
    if lower.shape != (len(T), len(buses)) or upper.shape != (len(T), len(buses)):
        raise ValueError("lower/upper must have shape (|T|, |bus_indices|)")

    # Create a subset for envelope-constrained buses
    model.B_env = pyo.Set(initialize=buses)

    low_map = {(b, t): float(lower[t_idx, buses.index(b)]) for t_idx, t in enumerate(T) for b in buses}
    up_map = {(b, t): float(upper[t_idx, buses.index(b)]) for t_idx, t in enumerate(T) for b in buses}

    model.p_env_lower = pyo.Param(model.B_env, model.T, initialize=low_map, mutable=False)
    model.p_env_upper = pyo.Param(model.B_env, model.T, initialize=up_map, mutable=False)

    if penalty is None:
        # Hard bounds
        def _ub_rule(m, b, t):
            return m.pg[b, t] <= m.p_env_upper[b, t]

        def _lb_rule(m, b, t):
            return m.pg[b, t] >= m.p_env_lower[b, t]

        model.p_env_ub = pyo.Constraint(model.B_env, model.T, rule=_ub_rule)
        model.p_env_lb = pyo.Constraint(model.B_env, model.T, rule=_lb_rule)
        return

    # Soft bounds with L1 slack
    model.p_env_pos = pyo.Var(model.B_env, model.T, domain=pyo.NonNegativeReals)
    model.p_env_neg = pyo.Var(model.B_env, model.T, domain=pyo.NonNegativeReals)

    def _ub_rule(m, b, t):
        return m.pg[b, t] <= m.p_env_upper[b, t] + m.p_env_pos[b, t]

    def _lb_rule(m, b, t):
        return m.pg[b, t] >= m.p_env_lower[b, t] - m.p_env_neg[b, t]

    model.p_env_ub = pyo.Constraint(model.B_env, model.T, rule=_ub_rule)
    model.p_env_lb = pyo.Constraint(model.B_env, model.T, rule=_lb_rule)

    coeff = float(penalty) * float(dt_hours)
    penalty_expr = coeff * sum(model.p_env_pos[b, t] + model.p_env_neg[b, t] for b in model.B_env for t in model.T)
    # Augment objective
    model.objective.expr = model.objective.expr + penalty_expr
