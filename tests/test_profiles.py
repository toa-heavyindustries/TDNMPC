"""Tests for synthetic profile generation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from profiles import (
    gen_load_profile,
    gen_pv_profile,
    gen_temp_profile,
    load_profiles,
    make_time_index,
    plot_profiles,
    save_profiles,
)


def test_make_time_index_returns_full_day() -> None:
    idx = make_time_index("2024-01-01", "1H")
    assert len(idx) == 24
    assert idx[0] == pd.Timestamp("2024-01-01T00:00:00")
    assert idx[-1] == pd.Timestamp("2024-01-01T23:00:00")


def test_save_and_load_profiles_roundtrip(tmp_path: Path) -> None:
    idx = make_time_index("2024-01-01", "30min")
    load = gen_load_profile(idx, seed=1)
    pv = gen_pv_profile(idx, peak_kw=500, seed=1)
    temp = gen_temp_profile(idx, seed=1)

    out_path = tmp_path / "profiles.csv"
    save_profiles(out_path, load=load, pv=pv, temp=temp)

    loaded = load_profiles(out_path)
    assert set(loaded.keys()) == {"load", "pv", "temp"}
    for key, original in {"load": load, "pv": pv, "temp": temp}.items():
        series = loaded[key]
        pd.testing.assert_index_equal(series.index, original.index)
        assert np.all(np.isfinite(series.values))


def test_plot_profiles_emits_png(tmp_path: Path) -> None:
    idx = make_time_index("2024-01-01", "1H")
    load = gen_load_profile(idx)
    pv = gen_pv_profile(idx)
    temp = gen_temp_profile(idx)

    out = tmp_path / "profiles.png"
    plot_profiles({"load": load, "pv": pv, "temp": temp}, out=out)
    assert out.exists()
    assert out.stat().st_size > 0

