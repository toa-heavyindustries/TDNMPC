"""Compute TI envelope from a run history and emit bounds + stats.

Usage:
  uv run python scripts/ti_envelope_from_history.py --run runs/demo --alpha 0.9 --margin 0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coord.ti_env import compute_envelope


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", dest="run_dir", type=Path, required=True, help="Run directory containing logs.csv")
    p.add_argument("--alpha", type=float, default=1.0, help="Shrink factor in (0,1], 1 keeps convex hull")
    p.add_argument("--margin", type=float, default=0.0, help="Non-negative envelope padding")
    return p.parse_args(argv)


def _activations(history: pd.DataFrame, env: dict[str, np.ndarray]) -> dict[str, Any]:
    # Use the next-step vectors (tso_vector/dso_vector) as proxy to count activations
    if "tso_vector" in history.columns:
        mat = np.vstack(history["tso_vector"].apply(np.asarray).to_list())
    elif "dso_vector" in history.columns:
        mat = np.vstack(history["dso_vector"].apply(np.asarray).to_list())
    else:
        return {"count": 0}

    lower = np.asarray(env["lower"]).reshape(1, -1) if env["lower"].ndim == 1 else env["lower"]
    upper = np.asarray(env["upper"]).reshape(1, -1) if env["upper"].ndim == 1 else env["upper"]
    L = lower[0, :]
    U = upper[0, :]
    eps = 1e-9
    hits_lo = (mat <= (L + eps)).sum()
    hits_up = (mat >= (U - eps)).sum()
    total = mat.size
    return {
        "samples": int(total),
        "hits_lower": int(hits_lo),
        "hits_upper": int(hits_up),
        "rate_lower": float(hits_lo / total) if total else 0.0,
        "rate_upper": float(hits_up / total) if total else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    hist_path = run_dir / "logs.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"History file not found: {hist_path}")

    df = pd.read_csv(hist_path)
    # Convert stringified lists to Python lists if necessary
    import ast
    for col in ("tso_vector", "dso_vector", "tso_h", "dso_h"):
        if col in df.columns:
            def _coerce(x):
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except Exception:
                        return x
                return x
            df[col] = df[col].apply(_coerce)
    bands = {"alpha": args.alpha, "margin": args.margin}
    env = compute_envelope(df, bands)
    (run_dir / "ti_env_bounds.json").write_text(json.dumps({
        "lower": env["lower"].tolist(),
        "upper": env["upper"].tolist(),
    }, indent=2))

    stats = _activations(df, env)
    (run_dir / "ti_env_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"bounds": str(run_dir / 'ti_env_bounds.json'), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
