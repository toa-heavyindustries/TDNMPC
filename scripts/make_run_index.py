"""Generate a simple Markdown index with links to plots and KPIs in a run dir."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir

    lines = [f"# Run: {run_dir.name}", ""]

    for name in [
        "summary.json",
        "report.json",
        "report.csv",
        "metrics.csv",
        "logs.csv",
        "lv_snapshots.summary.json",
        "timeseries.summary.json",
    ]:
        p = run_dir / name
        if p.exists():
            lines.append(f"- `{name}`")

    for img in sorted(run_dir.glob("*.png")):
        lines.append(f"- {img.name}")

    md = "\n".join(lines) + "\n"
    (run_dir / "INDEX.md").write_text(md)


if __name__ == "__main__":
    main()

