"""Generates and saves standard pandapower network cases as JSON."""

from __future__ import annotations

from pathlib import Path

import pandapower as pp
import pandapower.converter
import pandapower.networks as pn


def main() -> None:
    """Loads standard networks and saves them to the data directory."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Load IEEE 39-bus system
    print("Loading IEEE 39-bus network...")
    net39 = pn.case39()
    net39_path = data_dir / "ieee39.json"
    pp.to_json(net39, str(net39_path))
    print(f"Saved IEEE 39-bus network to {net39_path}")

    # Using IEEE 33-bus network for DSO as agreed.
    # The data/ieee33.json file is assumed to be present.
    print("Using IEEE 33-bus network for DSO as agreed.")

    print("\nBase network generation complete.")


if __name__ == "__main__":
    main()
