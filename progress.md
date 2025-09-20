# Project Progress Log

## Baseline Network Scaffold
- Added pandapower-based TSO case wrapper with boundary metadata (`src/tso/network.py`).
- Scaled CIGRE MV/LV feeder builder and metadata container (`src/dso/network.py`).
- Introduced baseline coupling planner for T–D experiments (`src/sim/base_networks.py`).
- Implemented composite T–D network assembly plus AC convergence test (`src/sim/base_networks.py`, `tests/test_base_networks.py`).
- Added deterministic config + logging scaffold (`configs/default.yaml`, `src/utils/random.py`, `src/utils/logging_utils.py`, `src/sim/runner.py`).
- Passed standalone TSO/DSO power-flow checks with tightened voltage limits (`src/tso/network.py`, `src/dso/network.py`, `src/sim/base_networks.py`).
- Built interface adapter utilities with tests (`src/interface/adapter.py`, `tests/test_interface_adapter.py`).
- Upgraded LinDistFlow sensitivities with finite-difference controls and MAPE validation (`src/models/lindistflow.py`, `tests/test_lindistflow.py`, `scripts/build_dso.py`).
- Integrated envelope pipeline with TSO bounds (`src/interface/envelope.py`, `scripts/build_envelope.py`, `src/opt/pyomo_tso.py`, `src/sim/runner.py`).
- Verified envelope-constrained TSO/DSO tracking step (`tests/test_closed_loop_tracking.py`).
- Added multi-step closed-loop simulator with forecast inputs & SoC KPIs (`src/sim/closed_loop.py`, `scripts/run_closed_loop.py`).
- Added tube-tightening metrics (sigma_p95, envelope shrink/noise hooks) to closed-loop flow (`src/sim/closed_loop.py`, `scripts/run_closed_loop.py`).
- Added Monte Carlo shrink sweep utility and test (`scripts/tune_shrink.py`, `tests/test_tune_shrink.py`).
