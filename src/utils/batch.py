"""Batch execution helpers for running multiple simulation seeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def run_batch(
    simulator: Callable[[int], dict[str, Any]],
    seeds: Sequence[int],
    run_dir: Path,
) -> pd.DataFrame:
    """Execute ``simulator`` for every seed, saving a consolidated CSV under ``run_dir``."""

    records: list[dict[str, Any]] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = simulator(int(seed))
        record = {"seed": int(seed), **result}
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(run_dir / "batch.csv", index=False)
    return frame
