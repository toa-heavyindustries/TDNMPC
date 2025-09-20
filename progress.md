# Project Progress Log

## Baseline Network Scaffold
- Added pandapower-based TSO case wrapper with boundary metadata (`src/tso/network.py`).
- Scaled CIGRE MV/LV feeder builder and metadata container (`src/dso/network.py`).
- Introduced baseline coupling planner for T–D experiments (`src/sim/base_networks.py`).
- Implemented composite T–D network assembly plus AC convergence test (`src/sim/base_networks.py`, `tests/test_base_networks.py`).
- Added deterministic config + logging scaffold (`configs/default.yaml`, `src/utils/random.py`, `src/utils/logging_utils.py`, `src/sim/runner.py`).
- Passed standalone TSO/DSO power-flow checks with tightened voltage limits (`src/tso/network.py`, `src/dso/network.py`, `src/sim/base_networks.py`).
- Built interface adapter utilities with tests (`src/interface/adapter.py`, `tests/test_interface_adapter.py`).
