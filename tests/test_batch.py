"""Tests for batch utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coord.admm import make_multi_dso
from utils.batch import run_batch


def test_make_multi_dso_generates_variants():
    base = np.array([1.0, 0.0])
    variants = make_multi_dso(3, base)
    assert len(variants) == 3
    assert all(var.shape == base.shape for var in variants)


def test_run_batch_creates_csv(tmp_path: Path) -> None:
    seeds = [1, 2, 3]

    def simulator(seed: int) -> dict[str, float]:
        return {"run_dir": str(tmp_path / f"seed_{seed}"), "final_residual": float(seed)}

    df = run_batch(simulator, seeds, tmp_path / "batch")
    assert (tmp_path / "batch" / "batch.csv").exists()
    assert len(df) == len(seeds)
