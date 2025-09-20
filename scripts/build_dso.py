"""Build a DSO network, export its data, and validate sensitivities.

Outputs:
- data/<case>.json: serialized pandapower net
- runs/<ts>/lin_check.json: summary pass/fail + mae/max
- runs/<ts>/sensitivity_eval.csv: per-sample error metrics (l2, max_abs)
- runs/<ts>/sensitivity_hist.png: histogram of max_abs error
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dso import ac_power_flow, build_ieee33, export_net
from models import linearize_lindistflow, validate_linearization
from utils import ensure_run_dir
import pandapower as pp

LOGGER = logging.getLogger("dso")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["ieee33"], default="ieee33", help="Network case")
    parser.add_argument("--out", type=Path, default=Path("data/ieee33.json"), help="Export path")
    parser.add_argument("--tag", default=None, help="Output run directory tag")
    parser.add_argument("--samples", type=int, default=20, help="Validation sample size")
    parser.add_argument("--tol", type=float, default=0.03, help="Validation tolerance")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    net = build_ieee33()
    LOGGER.info("Network built with %s buses and %s lines", len(net.bus), len(net.line))

    base_pf = ac_power_flow(net)
    LOGGER.info("Base power flow computed. Min voltage: %.3f pu", base_pf["vm_pu"].min())

    sens = linearize_lindistflow(net, base_pf)

    export_net(net, args.out)
    LOGGER.info("Network exported to %s", args.out)

    run_dir = ensure_run_dir(args.tag)
    metrics_path = run_dir / "lin_check.json"
    metrics = validate_linearization(net, sens, n_samples=args.samples, tol=args.tol)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Validation metrics written to %s", metrics_path)

    # Detailed sensitivity evaluation and histogram (per 基本实验.md)
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        pv_buses = net.load.bus.to_numpy()
        n = len(pv_buses)
        load_idx = net.load.index.to_numpy()

        base_p = net.load["p_mw"].to_numpy().copy()
        base_q = net.load["q_mvar"].to_numpy().copy()
        vm_base = sens["vm_base"]

        records = []
        K = max(1, int(args.samples))
        for k in range(K):
            dp = rng.normal(scale=0.01, size=n)
            dq = rng.normal(scale=0.01, size=n)

            net.load.loc[load_idx, "p_mw"] = base_p + dp
            net.load.loc[load_idx, "q_mvar"] = base_q + dq
            pp.runpp(net, algorithm="nr", recycle=None, numba=False)
            vm_ac = net.res_bus.vm_pu.loc[pv_buses].to_numpy()

            vm_lin = vm_base + sens["Rp"] @ dp + sens["Rq"] @ dq
            err = vm_ac - vm_lin
            records.append({
                "sample": k,
                "l2": float(np.linalg.norm(err)),
                "max_abs": float(np.max(np.abs(err))),
            })

        # Restore
        net.load.loc[load_idx, "p_mw"] = base_p
        net.load.loc[load_idx, "q_mvar"] = base_q
        pp.runpp(net, algorithm="nr", recycle=None, numba=False)

        df = pd.DataFrame(records)
        df.to_csv(run_dir / "sensitivity_eval.csv", index=False)
        LOGGER.info("Detailed sensitivity evaluation saved: %s", run_dir / "sensitivity_eval.csv")

        # Histogram of max_abs
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["max_abs"], bins=20, color="#4C78A8", alpha=0.85)
        ax.set_title("Sensitivity Max-Abs Error")
        ax.set_xlabel("|ΔV_ac - ΔV_lin|")
        ax.set_ylabel("count")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "sensitivity_hist.png")
        plt.close(fig)
        LOGGER.info("Histogram saved: %s", run_dir / "sensitivity_hist.png")
    except Exception as exc:
        LOGGER.warning("Detailed sensitivity eval failed: %s", exc)


if __name__ == "__main__":
    main()
