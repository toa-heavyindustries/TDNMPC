"""Synthetic profile generation, persistence, and visualization utilities."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_time_index(date: str, freq: str) -> pd.DatetimeIndex:
    """Return a 24-hour time index starting at ``date`` using the provided frequency.

    Parameters
    ----------
    date:
        ISO-like date string (e.g. ``"2024-01-01"``) used as the inclusive start.
    freq:
        Pandas frequency alias (e.g. ``"5min"``, ``"1H"``).

    Returns
    -------
    pd.DatetimeIndex
        Datetime index covering one day with left-closed, right-open convention.
    """

    start = pd.Timestamp(date)
    step = pd.tseries.frequencies.to_offset(freq)
    if step is None:
        raise ValueError(f"Invalid frequency: {freq}")

    periods = int(pd.Timedelta(hours=24) / step.delta)
    if periods <= 0:
        raise ValueError("Frequency must produce at least one period within 24h")

    return pd.date_range(start=start, periods=periods, freq=freq, inclusive="left")


def gen_load_profile(idx: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    """Generate a stylised residential load profile.

    Parameters
    ----------
    idx:
        Datetime index representing the simulation horizon.
    seed:
        Seed fed into ``numpy.random.default_rng`` for reproducibility.

    Returns
    -------
    pd.Series
        Load in kW aligned to ``idx`` with no missing values.
    """

    rng = np.random.default_rng(seed)
    hours = idx.hour + idx.minute / 60

    morning_peak = 0.35 * np.exp(-0.5 * ((hours - 7) / 1.5) ** 2)
    evening_peak = 0.55 * np.exp(-0.5 * ((hours - 20) / 2) ** 2)
    base = 0.4 + 0.15 * np.sin(2 * np.pi * (hours - 13) / 24)
    noise = rng.normal(0, 0.03, size=len(idx))

    profile = np.maximum(base + morning_peak + evening_peak + noise, 0.05)
    series = pd.Series(profile * 1000, index=idx, name="load")
    return series


def gen_pv_profile(idx: pd.DatetimeIndex, peak_kw: float = 1000.0, seed: int = 42) -> pd.Series:
    """Create a bell-shaped PV production curve with stochastic cloud attenuation.

    Parameters
    ----------
    idx:
        Datetime index for the horizon.
    peak_kw:
        Maximum AC output of the PV plant in kW.
    seed:
        Random seed used for cloud attenuation noise.

    Returns
    -------
    pd.Series
        PV generation in kW (non-negative) indexed by ``idx``.
    """

    rng = np.random.default_rng(seed + 1)
    hours = idx.hour + idx.minute / 60
    solar_elevation = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None)
    cloud_factor = 0.9 + 0.1 * np.sin(4 * np.pi * hours / 24)
    noise = rng.normal(0, 0.05, size=len(idx))

    generation = np.maximum(peak_kw * solar_elevation * cloud_factor * (1 + noise), 0.0)
    return pd.Series(generation, index=idx, name="pv")


def gen_temp_profile(idx: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    """Generate an outdoor temperature trajectory with diurnal dynamics.

    Parameters
    ----------
    idx:
        Datetime index for the horizon.
    seed:
        Random seed controlling temperature perturbations.

    Returns
    -------
    pd.Series
        Ambient temperature in degrees Celsius referenced to ``idx``.
    """

    rng = np.random.default_rng(seed + 2)
    hours = idx.hour + idx.minute / 60
    base_temp = 20 + 5 * np.sin(2 * np.pi * (hours - 15) / 24)
    noise = rng.normal(0, 0.8, size=len(idx))
    return pd.Series(base_temp + noise, index=idx, name="temp")


def save_profiles(path: Path, **series: pd.Series) -> None:
    """Persist multiple profiles into a single CSV file.

    Parameters
    ----------
    path:
        Target CSV file path. Parent directories are created automatically.
    **series:
        Named pandas Series sharing the same index.
    """

    if not series:
        raise ValueError("At least one series must be provided")

    index = _validate_common_index(series)
    df = pd.DataFrame({name: s.reindex(index).fillna(0.0) for name, s in series.items()})
    df.insert(0, "time", index)
    df = df.fillna(0.0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_profiles(path: Path) -> dict[str, pd.Series]:
    """Load profiles previously saved by :func:`save_profiles`.

    Parameters
    ----------
    path:
        CSV file path to load.

    Returns
    -------
    dict[str, pd.Series]
        Mapping from column name to series aligned on the original time index.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Profile file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.set_index("time")
    df.index.name = None
    return {col: df[col].copy() for col in df.columns}


def plot_profiles(series: dict[str, pd.Series], out: Path | None = None) -> None:
    """Plot multiple profiles on a single axis and optionally save to disk.

    Parameters
    ----------
    series:
        Mapping of label to pandas Series. Index alignment is handled automatically.
    out:
        Optional output path. When provided, the plot is written to disk and the figure
        is closed. When omitted, the figure is left open for interactive backends.
    """

    if not series:
        raise ValueError("No profiles provided for plotting")

    aligned = _align_series(series)

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, ser in aligned.items():
        ax.plot(ser.index, ser.values, label=name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


def _validate_common_index(series: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """Verify that all series share the same datetime index."""

    indexes = {name: s.index for name, s in series.items()}
    first_index: pd.DatetimeIndex | None = None
    for name, idx in indexes.items():
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError(f"Series '{name}' must have a DatetimeIndex")
        if first_index is None:
            first_index = idx
            continue
        if not first_index.equals(idx):
            raise ValueError("All series must share the same index")
    return first_index if first_index is not None else pd.DatetimeIndex([])


def _align_series(series: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Reindex the input series onto the union of their indices."""

    # Combine indexes and forward fill missing values to avoid NaNs in visualisation.
    union_index = pd.DatetimeIndex(sorted({ts for s in series.values() for ts in s.index}))
    aligned: dict[str, pd.Series] = {}
    for name, ser in series.items():
        aligned[name] = (
            ser.reindex(union_index)
            .interpolate(method="time")
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        aligned[name].name = name
    return aligned

