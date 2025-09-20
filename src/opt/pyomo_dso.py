"""Pyomo-based DSO economic dispatch under LinDistFlow sensitivities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

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

    load_dict = {(b, t): float(load_matrix[b, t]) for b in range(n_bus) for t in range(steps)}
    pv_dict = {(b, t): float(pv_matrix[b, t]) for b in range(n_bus) for t in range(steps)}
    vm_base_dict = {b: float(vm_base[b]) for b in range(n_bus)}
    Rp_dict = {(b, j): float(Rp[b, j]) for b in range(n_bus) for j in range(n_bus)}
    Rq_dict = {(b, j): float(Rq[b, j]) for b in range(n_bus) for j in range(n_bus)}

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

