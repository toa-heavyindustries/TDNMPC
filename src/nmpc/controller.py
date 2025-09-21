"""Simplified NMPC controller coordinating boundary consensus via ADMM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from coord.admm import ADMMConfig, run_admm
from coord.interface import coupling_residuals
from coord.ti_env import Envelope
from nmpc.base import BaseController, EnvelopeConfig

TSOSolver = Callable[[np.ndarray], tuple[np.ndarray, Any]]
DSOSolver = Callable[[np.ndarray], tuple[np.ndarray, Any]]


@dataclass
class NMPCConfig:
    """Configuration for the NMPC controller.

    Attributes
    ----------
    size:
        Dimension of the coupled boundary vector.
    admm:
        ADMM configuration controlling iterations and tolerances.
    tso_solver:
        Callable receiving the vector ``z - u`` (TSO perspective) and returning a tuple
        ``(boundary_flow, metadata)``.
    dso_solver:
        Callable receiving the vector ``x + u`` (DSO perspective) and returning a tuple
        ``(boundary_power, metadata)``.
    envelope_margin:
        Margin used when updating the trajectory envelope.
    envelope_alpha:
        Exponential smoothing factor for envelope updates.
    """

    size: int
    admm: ADMMConfig
    tso_solver: TSOSolver
    dso_solver: DSOSolver
    envelope_margin: float = 0.05
    envelope_alpha: float = 0.3


@dataclass
class NMPCStepResult:
    """Outputs from a single NMPC iteration."""

    tso_vector: np.ndarray
    dso_vector: np.ndarray
    residuals: dict[str, float]
    admm_history: list[dict[str, float]]
    envelope: Envelope
    tso_metadata: Any
    dso_metadata: Any


class NMPCController(BaseController):
    """High-level orchestrator executing a single ADMM-based NMPC step."""

    def __init__(self, config: NMPCConfig):
        envelope_cfg = EnvelopeConfig(
            size=config.size,
            margin=config.envelope_margin,
            alpha=config.envelope_alpha,
        )
        super().__init__(envelope_cfg)
        self.config = config

    def run_step(self, initial_guess: np.ndarray | None = None) -> NMPCStepResult:
        size = self.config.size
        if initial_guess is None:
            initial_guess = np.zeros(size, dtype=float)
        else:
            initial_guess = np.asarray(initial_guess, dtype=float)
            if initial_guess.shape[0] != size:
                raise ValueError("Initial guess size mismatch")

        last_tso_meta: Any = None
        last_dso_meta: Any = None

        def tso_update(v: np.ndarray) -> np.ndarray:
            nonlocal last_tso_meta
            value, meta = self.config.tso_solver(np.asarray(v, dtype=float))
            if value.shape[0] != size:
                raise ValueError("TSO solver returned wrong dimension")
            last_tso_meta = meta
            return value

        def dso_update(v: np.ndarray) -> np.ndarray:
            nonlocal last_dso_meta
            value, meta = self.config.dso_solver(np.asarray(v, dtype=float))
            if value.shape[0] != size:
                raise ValueError("DSO solver returned wrong dimension")
            last_dso_meta = meta
            return value

        admm_out = run_admm(tso_update, dso_update, self.config.admm)

        tso_vec = np.asarray(admm_out["x"], dtype=float)
        dso_vec = np.asarray(admm_out["z"], dtype=float)
        tso_meta = last_tso_meta
        dso_meta = last_dso_meta

        residual = coupling_residuals(tso_vec, dso_vec)
        self.update_envelope(dso_vec)

        return NMPCStepResult(
            tso_vector=tso_vec,
            dso_vector=dso_vec,
            residuals=residual,
            admm_history=admm_out["history"],
            envelope=self.envelope,
            tso_metadata=tso_meta,
            dso_metadata=dso_meta,
        )
