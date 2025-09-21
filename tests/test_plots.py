"""Tests for visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from viz.plots import plot_convergence, plot_timeseries


def test_plot_timeseries_writes_png(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 0.4, 0.3]})
    out = tmp_path / "timeseries.png"
    plot_timeseries(df, ["a", "b"], out)
    assert out.exists()


def test_plot_convergence_requires_columns(tmp_path: Path) -> None:
    hist = pd.DataFrame({"primal_residual": [1.0, 0.1], "dual_residual": [1.0, 0.2]})
    out = tmp_path / "conv.png"
    plot_convergence(hist, out)
    assert out.exists()
