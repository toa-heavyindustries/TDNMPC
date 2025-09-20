"""Linearized distribution power flow utilities for the IEEE feeders."""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pandas as pd


def linearize_lindistflow(net: pp.pandapowerNet, base: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute sensitivity matrices for active/reactive injections to bus voltage magnitudes."""

    load_indices = net.load.index.to_numpy()
    pv_buses = net.load.bus.to_numpy()
    n_load = len(pv_buses)

    vm_base = base.loc[pv_buses, "vm_pu"].to_numpy()

    epsilon = 1e-4
    sensitivities: list[np.ndarray] = []
    for var in ["p_mw", "q_mvar"]:
        jac = np.zeros((n_load, n_load))
        for col, load_idx in enumerate(load_indices):
            original = net.load.at[load_idx, var]
            net.load.at[load_idx, var] = original + epsilon
            pp.runpp(net, algorithm="nr", recycle=None, numba=False)
            vm = net.res_bus.vm_pu.loc[pv_buses].to_numpy()
            jac[:, col] = (vm - vm_base) / epsilon
            net.load.at[load_idx, var] = original
        sensitivities.append(jac)

    pp.runpp(net, algorithm="nr", recycle=None, numba=False)

    return {"Rp": sensitivities[0], "Rq": sensitivities[1], "vm_base": vm_base}


def predict_voltage_delta(sens: dict[str, np.ndarray], dp: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Predict voltage deviations using sensitivity matrices and power perturbations."""

    Rp = sens["Rp"]
    Rq = sens["Rq"]
    return Rp @ dp + Rq @ dq


def validate_linearization(
    net: pp.pandapowerNet,
    sens: dict[str, np.ndarray],
    n_samples: int = 20,
    tol: float = 0.03,
) -> dict[str, float]:
    """Validate the linearization by sampling random perturbations."""

    rng = np.random.default_rng(42)
    pv_buses = net.load.bus.to_numpy()
    vm_base = sens["vm_base"]

    errors: list[float] = []
    max_errors: list[float] = []
    load_indices = net.load.index.to_numpy()

    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()

    for _ in range(n_samples):
        dp = rng.normal(scale=0.01, size=len(pv_buses))
        dq = rng.normal(scale=0.01, size=len(pv_buses))

        net.load.loc[load_indices, "p_mw"] = base_p + dp
        net.load.loc[load_indices, "q_mvar"] = base_q + dq

        pp.runpp(net, algorithm="nr", recycle=None, numba=False)
        vm_actual = net.res_bus.vm_pu.loc[pv_buses].to_numpy()
        vm_pred = vm_base + predict_voltage_delta(sens, dp, dq)

        diff = vm_actual - vm_pred
        errors.append(float(np.mean(np.abs(diff))))
        max_errors.append(float(np.max(np.abs(diff))))

    net.load.loc[load_indices, "p_mw"] = base_p
    net.load.loc[load_indices, "q_mvar"] = base_q
    pp.runpp(net, algorithm="nr", recycle=None, numba=False)

    mae = float(np.mean(errors))
    mmax = float(np.max(max_errors))
    return {"mae": mae, "max": mmax, "passed": mae <= tol and mmax <= 2 * tol}

