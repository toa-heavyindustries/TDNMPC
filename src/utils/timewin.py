"""Time horizon and window management utilities for NMPC scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class Horizon:
    """Time horizon specification for NMPC problems."""

    start: pd.Timestamp
    steps: int
    dt_min: int

    @property
    def end(self) -> pd.Timestamp:
        """Return the exclusive end timestamp of the horizon."""

        delta = pd.Timedelta(minutes=self.steps * self.dt_min)
        return self.start + delta


def make_horizon(start: str, steps: int, dt_min: int) -> Horizon:
    """Construct a :class:`Horizon` from string inputs."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if dt_min <= 0:
        raise ValueError("dt_min must be positive")

    ts = pd.Timestamp(start)
    return Horizon(start=ts, steps=int(steps), dt_min=int(dt_min))


def rolling_windows(idx: pd.DatetimeIndex, steps: int) -> list[pd.DatetimeIndex]:
    """Generate non-overlapping rolling windows over the provided index."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    total = len(idx)
    windows: list[pd.DatetimeIndex] = []
    for offset in range(0, total, steps):
        window = idx[offset : offset + steps]
        if len(window) == steps:
            windows.append(window)
    return windows


def align_profiles(idx: pd.DatetimeIndex, profiles: dict[str, pd.Series]) -> pd.DataFrame:
    """Align named series to the provided index, forward/back filling gaps."""

    if not profiles:
        raise ValueError("No profiles provided")

    data = {}
    for name, series in profiles.items():
        aligned = series.reindex(idx).interpolate(method="time").ffill().bfill()
        data[name] = aligned
    df = pd.DataFrame(data, index=idx)
    df.index.name = "time"
    return df

