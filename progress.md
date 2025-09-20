# Project Progress Log

## Baseline Network Scaffold
- Added pandapower-based TSO case wrapper with boundary metadata (`src/tso/network.py`).
- Scaled CIGRE MV/LV feeder builder and metadata container (`src/dso/network.py`).
- Introduced baseline coupling planner for T–D experiments (`src/sim/base_networks.py`).
- Implemented composite T–D network assembly plus AC convergence test (`src/sim/base_networks.py`, `tests/test_base_networks.py`).
