"""Global random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed common RNG backends for reproducibility."""

    if seed < 0:
        raise ValueError("Seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(bool(deterministic))


def seed_sequence(base_seed: int, count: int) -> list[int]:
    """Generate a reproducible sequence of integer seeds."""

    if count < 0:
        raise ValueError("count must be non-negative")
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(low=0, high=2**31 - 1)) for _ in range(count)]
