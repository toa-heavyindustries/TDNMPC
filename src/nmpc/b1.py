"""Centralized baseline controller (B1): one-shot DSO then TSO solve.

Produces an NMPCStepResult-compatible structure so it can reuse logging/plots.
"""

from __future__ import annotations

import numpy as np

from nmpc.base import BaseController, EnvelopeConfig
from nmpc.controller import NMPCConfig, NMPCStepResult


class B1Controller(BaseController):
    """Centralized controller that solves DSOs first, then the TSO."""

    def __init__(self, config: NMPCConfig):
        envelope_cfg = EnvelopeConfig(
            size=config.size,
            margin=config.envelope_margin,
            alpha=config.envelope_alpha,
        )
        super().__init__(envelope_cfg)
        self.config = config

    def run_step(self) -> NMPCStepResult:
        """Run a single step of the centralized B1 algorithm."""
        # 1. Compute local DSO response ignoring TSO signal
        dso_vec, dso_meta = self.config.dso_solver(np.zeros(self.size))

        # 2. Feed DSO response to TSO as fixed boundary targets
        tso_vec, tso_meta = self.config.tso_solver(dso_vec)

        # 3. Update envelope and craft result
        self.update_envelope(dso_vec)

        return NMPCStepResult(
            tso_vector=tso_vec,
            dso_vector=dso_vec,
            residuals={
                "max": float(np.max(np.abs(tso_vec - dso_vec))),
                "mean": float(np.mean(np.abs(tso_vec - dso_vec))),
                "l2": float(np.linalg.norm(tso_vec - dso_vec)),
            },
            admm_history=[],
            envelope=self.envelope,
            tso_metadata=tso_meta,
            dso_metadata=dso_meta,
        )
