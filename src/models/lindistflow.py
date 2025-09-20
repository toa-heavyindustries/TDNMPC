"""Linearized distribution power flow utilities for the IEEE feeders."""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pandas as pd


def linearize_lindistflow(
    net: pp.pandapowerNet,
    base: pd.DataFrame,
    *,
    perturbation_mw: float = 0.5,
    perturbation_mvar: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute sensitivity matrices for active/reactive injections to bus voltage magnitudes."""

    load_indices = net.load.index.to_numpy()
    pv_buses = net.load.bus.to_numpy()
    n_load = len(pv_buses)

    vm_base = base.loc[pv_buses, "vm_pu"].to_numpy()

    dq_step = perturbation_mvar if perturbation_mvar is not None else perturbation_mw

    def _central_difference(var: str, step: float) -> np.ndarray:
        jac = np.zeros((n_load, n_load))
        for col, load_idx in enumerate(load_indices):
            original = float(net.load.at[load_idx, var])
            net.load.at[load_idx, var] = original + step
            pp.runpp(net, algorithm="nr", recycle=None, numba=False)
            vm_plus = net.res_bus.vm_pu.loc[pv_buses].to_numpy()

            net.load.at[load_idx, var] = original - step
            pp.runpp(net, algorithm="nr", recycle=None, numba=False)
            vm_minus = net.res_bus.vm_pu.loc[pv_buses].to_numpy()

            jac[:, col] = (vm_plus - vm_minus) / (2.0 * step)
            net.load.at[load_idx, var] = original
        return jac

    Rp = _central_difference("p_mw", max(perturbation_mw, 1e-4))
    Rq = _central_difference("q_mvar", max(dq_step, 1e-4))

    pp.runpp(net, algorithm="nr", recycle=None, numba=False)

    return {"Rp": Rp, "Rq": Rq, "vm_base": vm_base}


def predict_voltage_delta(sens: dict[str, np.ndarray], dp: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Predict voltage deviations using sensitivity matrices and power perturbations."""

    Rp = sens["Rp"]
    Rq = sens["Rq"]
    return Rp @ dp + Rq @ dq


def validate_linearization(
    net: pp.pandapowerNet,
    sens: dict[str, np.ndarray],
    n_samples: int = 50,
    tol: float = 0.03,
    *,
    max_delta_mw: float = 0.5,
    max_delta_mvar: float | None = None,
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

    dq_max = max_delta_mvar if max_delta_mvar is not None else max_delta_mw

    for _ in range(n_samples):
        dp = rng.uniform(-max_delta_mw, max_delta_mw, size=len(pv_buses))
        dq = rng.uniform(-dq_max, dq_max, size=len(pv_buses))

        net.load.loc[load_indices, "p_mw"] = base_p + dp
        net.load.loc[load_indices, "q_mvar"] = base_q + dq

        pp.runpp(net, algorithm="nr", recycle=None, numba=False)
        vm_actual = net.res_bus.vm_pu.loc[pv_buses].to_numpy()
        vm_pred = vm_base + predict_voltage_delta(sens, dp, dq)

        diff = vm_actual - vm_pred
        abs_base = np.maximum(vm_actual, 1e-6)
        mape = np.abs(diff) / abs_base
        errors.append(float(np.mean(mape)))
        max_errors.append(float(np.max(np.abs(diff))))

    net.load.loc[load_indices, "p_mw"] = base_p
    net.load.loc[load_indices, "q_mvar"] = base_q
    pp.runpp(net, algorithm="nr", recycle=None, numba=False)

    mae = float(np.mean(errors))
    mmax = float(np.max(max_errors))
    return {"mape": mae, "max_abs": mmax, "passed": mae <= tol and mmax <= 2 * tol}
