"""Greedy baseline controller (B0): local DSO-only adjustment without coordination.

Produces an NMPCStepResult-compatible structure so it can reuse logging/plots.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nmpc.base import BaseController, EnvelopeConfig
from nmpc.controller import NMPCStepResult

DSOSolver = Callable[[np.ndarray], tuple[np.ndarray, Any]]


@dataclass
class GreedyConfig:
    size: int
    dso_solver: DSOSolver
    envelope_margin: float = 0.05
    envelope_alpha: float = 0.3


class GreedyController(BaseController):
    def __init__(self, config: GreedyConfig):
        envelope_cfg = EnvelopeConfig(
            size=config.size,
            margin=config.envelope_margin,
            alpha=config.envelope_alpha,
        )
        super().__init__(envelope_cfg)
        self.config = config

    def run_step(self) -> NMPCStepResult:
        # Greedy: ignore TSO signals, just compute local DSO action once.
        zero = np.zeros(self.size, dtype=float)
        dso_vec, dso_meta = self.config.dso_solver(zero)
        tso_vec = dso_vec.copy()

        self.update_envelope(dso_vec)

        return NMPCStepResult(
            tso_vector=tso_vec,
            dso_vector=dso_vec,
            residuals={"max": 0.0, "mean": 0.0, "l2": 0.0},
            admm_history=[],
            envelope=self.envelope,
            tso_metadata={},
            dso_metadata=dso_meta,
        )
