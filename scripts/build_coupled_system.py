"""Builds the coupled TSO-DSO system based on IEEE-39 and IEEE-33 networks."""

from __future__ import annotations

from pathlib import Path

import pandapower as pp
import pandapower.networks as pn


def main() -> None:
    """Loads TSO and DSO networks and connects them with transformers."""
    data_dir = Path("data")
    output_dir = Path("data") # Save combined net in data folder for now
    output_dir.mkdir(exist_ok=True)

    # 1. Load TSO and DSO networks
    print("Loading TSO (IEEE-39) network...")
    net_tso = pp.from_json(str(data_dir / "ieee39.json"))
    print(f"TSO network has {len(net_tso.bus)} buses.")

    print("Loading DSO (IEEE-33) network...")
    net_dso_template = pp.from_json(str(data_dir / "ieee33.json"))
    print(f"DSO template network has {len(net_dso_template.bus)} buses.")
    print(f"Type of net_dso_template: {type(net_dso_template)}")

    # 2. Define connection points
    # TSO buses to connect DSOs to (from 基座环境.md)
    tso_connection_buses = [16, 18, 21]
    # DSO bus to connect to TSO (assuming bus 0 for IEEE-33)
    dso_pcc_bus = 0

    # Transformer parameters (from 基座环境.md)
    # First stage: 345kV (TSO) to 115kV (intermediate)
    trafo_345_115_sn_mva = 100 # Example rating, adjust as needed
    trafo_345_115_vk_percent = 8 # Example impedance
    trafo_345_115_vkr_percent = 0.5 # Example impedance

    # Initialize the combined network with the TSO network
    net_combined = net_tso

    # Define standard type for 345/115 kV transformer
    pp.create_std_type(net=net_combined, data={
        "sn_mva": trafo_345_115_sn_mva,
        "vn_hv_kv": 345.0,
        "vn_lv_kv": 115.0,
        "vk_percent": trafo_345_115_vk_percent,
        "vkr_percent": trafo_345_115_vkr_percent,
        "pfe_kw": 0,
        "i0_percent": 0,
        "shift_degree": 0.0 # Added missing parameter
    }, element="trafo", name="345_115_custom")

    new_dso_bus_indices = [] # To store the new bus indices of the merged DSOs

    # 3. Loop through TSO connection buses and add DSOs
    for i, tso_bus_idx in enumerate(tso_connection_buses):
        print(f"\nConnecting DSO {i+1} to TSO bus {tso_bus_idx}...")
        # Create a copy of the DSO template for each connection
        net_dso = net_dso_template.copy()

        # Step 1: Add 345/115 kV Transformer and Intermediate Bus
        intermediate_bus_115kv = pp.create_bus(net_combined, name=f"TSO_Bus_{tso_bus_idx}_to_DSO_{i+1}_115kV", vn_kv=115.0)
        pp.create_transformer(net=net_combined, hv_bus=tso_bus_idx, lv_bus=intermediate_bus_115kv, std_type="345_115_custom")

        # Step 2: Add 115/12.47 kV Transformer and PCC Bus
        pcc_bus_12kv = pp.create_bus(net_combined, name=f"DSO_{i+1}_PCC_12kV", vn_kv=12.47)
        pp.create_transformer(net=net_combined, hv_bus=intermediate_bus_115kv, lv_bus=pcc_bus_12kv, std_type="115_12_custom")

        # Step 3: Merge DSO Network
        # The ppc_bus argument is the bus in net_b (net_dso) that connects to net_a (net_combined)
        # The net_b_pcc_bus argument is the bus in net_a (net_combined) that net_b connects to
        pp.merge_nets(net_combined, net_dso, ppc_bus=net_dso.bus.index[dso_pcc_bus], net_b_pcc_bus=pcc_bus_12kv)

        # Placeholder for merged DSO's PCC bus
        # This will be the bus in net_combined that the DSO's original PCC bus maps to
        # (This line is now redundant as merge_net handles the connection directly)
        # merged_dso_pcc_bus_in_combined_net = -1 # To be determined after merging

        # --- Capacity Adjustment Logic will go here ---

    # 4. Save the combined network
    combined_net_path = output_dir / "coupled_tso_dso_net.json"
    pp.to_json(net_combined, str(combined_net_path))
    print(f"\nCombined TSO-DSO network saved to {combined_net_path}")


if __name__ == "__main__":
    main()
