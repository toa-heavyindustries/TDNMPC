"""Shared helpers for NMPC controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coord.ti_env import Envelope, create_envelope, update_envelope


@dataclass(frozen=True)
class EnvelopeConfig:
    size: int
    margin: float = 0.05
    alpha: float = 0.3


class BaseController:
    """Provide common envelope handling logic for controller strategies."""

    def __init__(self, envelope_cfg: EnvelopeConfig) -> None:
        if envelope_cfg.size <= 0:
            raise ValueError("envelope_cfg.size must be positive")
        self._size = envelope_cfg.size
        self.envelope = create_envelope(
            size=envelope_cfg.size,
            margin=envelope_cfg.margin,
            alpha=envelope_cfg.alpha,
        )

    @property
    def size(self) -> int:
        return self._size

    def update_envelope(self, values: np.ndarray) -> Envelope:
        """Update the internal envelope with new boundary values."""

        self.envelope = update_envelope(self.envelope, values)
        return self.envelope
