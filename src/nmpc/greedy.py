"""Greedy baseline controller (B0): local DSO-only adjustment without coordination.

Produces an NMPCStepResult-compatible structure so it can reuse logging/plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from coord.ti_env import Envelope, create_envelope, update_envelope
from nmpc.controller import NMPCStepResult


DSOSolver = Callable[[np.ndarray], tuple[np.ndarray, Any]]


@dataclass
class GreedyConfig:
    size: int
    dso_solver: DSOSolver
    envelope_margin: float = 0.05
    envelope_alpha: float = 0.3


class GreedyController:
    def __init__(self, config: GreedyConfig):
        if config.size <= 0:
            raise ValueError("config.size must be positive")
        self.config = config
        self.envelope: Envelope = create_envelope(
            size=config.size, margin=config.envelope_margin, alpha=config.envelope_alpha
        )

    def run_step(self) -> NMPCStepResult:
        # Greedy: ignore TSO signals, just compute local DSO action once.
        zero = np.zeros(self.config.size, dtype=float)
        dso_vec, dso_meta = self.config.dso_solver(zero)
        tso_vec = dso_vec.copy()

        self.envelope = update_envelope(self.envelope, dso_vec)

        return NMPCStepResult(
            tso_vector=tso_vec,
            dso_vector=dso_vec,
            residuals={"max": 0.0, "mean": 0.0, "l2": 0.0},
            admm_history=[],
            envelope=self.envelope,
            tso_metadata={},
            dso_metadata=dso_meta,
        )

