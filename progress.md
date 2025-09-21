# Project Progress Log

## 2025-09-21
- Add CLI entrypoint `src/app/main.py`; `codex-nmpc --config` now runs scenarios.
- Code quality: pass ruff checks; update `pyproject.toml` with line-length=120, ignore `E501`, and per-file ignore `C901` for `src/sim/runner.py`.
- Fix missing `strict` in `zip(...)`: `src/viz/plots.py`, `tests/test_timewin.py`.
- Remove/organize unused imports/vars: `tests/test_plots.py`, `tests/test_sim_runner.py`, `scripts/plot_closed_loop.py`, `tests/test_envelope.py`.
- Type fix in `src/sim/runner.py` fallback: `_pd.DataFrame` -> `pd.DataFrame`.
- Tidy long lines and unused in `scripts/build_coupled_system.py`; add experimental note.
- Tests: `pytest -q` passing.

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

## Solver Robustness & Controller Refactor
- Added shared Pyomo dict utility and tightened solver status checks (`src/utils/pyomo_utils.py`, `src/opt/pyomo_tso.py`, `src/opt/pyomo_dso.py`).
- Logged TSO/DSO solver failures with safe fallbacks and cleaned controller factory (`src/sim/runner.py`).
- Introduced reusable controller base class to unify envelope handling (`src/nmpc/base.py`, `src/nmpc/controller.py`, `src/nmpc/greedy.py`).
- Relocated batch runner to utilities and updated callers (`src/utils/batch.py`, `src/coord/admm.py`, `src/coord/__init__.py`, `scripts/run_experiment.py`, `tests/test_batch.py`).
- Synced README run instructions and module references with current layout (`README.md`).
