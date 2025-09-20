"""Generate synthetic load/PV/temperature profiles and persist them to disk."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from profiles import (
    gen_load_profile,
    gen_pv_profile,
    gen_temp_profile,
    make_time_index,
    plot_profiles,
    save_profiles,
)
from utils import ensure_run_dir

LOGGER = logging.getLogger("profiles")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the profile generator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=pd.Timestamp.utcnow().date().isoformat(), help="ISO date")
    parser.add_argument("--freq", default="5min", help="Pandas frequency alias (default: 5min)")
    parser.add_argument("--peak-kw", type=float, default=1000.0, help="PV peak power in kW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=Path, default=Path("data/profiles.csv"), help="Output CSV path")
    parser.add_argument("--tag", default=None, help="Optional run directory tag")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (useful for headless environments).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the profile generation script."""

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    idx = make_time_index(args.date, args.freq)
    LOGGER.info("Time index generated: %s points (%s to %s)", len(idx), idx[0], idx[-1])

    load = gen_load_profile(idx, seed=args.seed)
    pv = gen_pv_profile(idx, peak_kw=args.peak_kw, seed=args.seed)
    temp = gen_temp_profile(idx, seed=args.seed)

    save_profiles(args.out, load=load, pv=pv, temp=temp)
    LOGGER.info("Profiles saved to %s", args.out)

    if not args.no_plot:
        run_dir = ensure_run_dir(args.tag)
        plot_path = run_dir / "profiles.png"
        plot_profiles({"load": load, "pv": pv, "temp": temp}, out=plot_path)
        LOGGER.info("Profile plot written to %s", plot_path)


if __name__ == "__main__":
    main()

