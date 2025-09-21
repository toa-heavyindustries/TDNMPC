# Project Progress Log

- 2025-09-21 运行脚本补全：新增 ADMM CLI
  - 新增文件：scripts/run_admm.py
  - 功能：基于现有 sim.scenario 构建（toy 或模型模式），运行共识 ADMM，输出收敛历史（CSV）与曲线（PNG）。
  - 产物：runs/<tag>/admm_history.csv, admm_convergence.png, admm_summary.json
  - 用法示例：
    - uv run python scripts/run_admm.py --cfg config/demo.yaml --tag admm_demo
  - 关联规划：对应 README Step 8（ADMM），为后续 TI-NMPC 实验提供独立收敛验证入口。

- 2025-09-21 设备与控制增强：接口分接控制与电容步进
  - 更新文件：scripts/exp_devices.py
  - 新增：尝试附加 DiscreteTapControl（若可用），并加入贪心分接调谐；将并联电容改为分步 create_shunt_as_capacitor（3×2 MVAr），并提供逐步使能逻辑。
  - 参数：--bess-scale, --tap-target, --tap-tol

- 2025-09-21 鲁棒性扫参脚本
  - 新增文件：scripts/exp_robust_sweep.py
  - 维度：变压器容量（25/40 MVA）、BESS 比例（0.5/1.0/1.5）、电压带（0.95–1.05 / 0.96–1.04）、tap 容差（0.005/0.01）
  - 产物：runs/<tag>/robust_sweep.json

- 2025-09-21 TI 包络离线计算脚本
  - 新增文件：scripts/ti_envelope_from_history.py
  - 功能：从 runs/<tag>/logs.csv 读取步进历史，按 alpha/margin 计算 time-varying bounds，输出 ti_env_bounds.json 与命中统计 ti_env_stats.json。
  - 用法：uv run python scripts/ti_envelope_from_history.py --run runs/demo --alpha 0.9 --margin 0.0

## 2025-09-21
- Add CLI entrypoint `src/app/main.py`; prefer `uv run main --config <name>` (alias: `uv run codex-nmpc`).
- Code quality: pass ruff checks; update `pyproject.toml` with line-length=120, ignore `E501`, and per-file ignore `C901` for `src/sim/runner.py`.
- Fix missing `strict` in `zip(...)`: `src/viz/plots.py`, `tests/test_timewin.py`.
- Remove/organize unused imports/vars: `tests/test_plots.py`, `tests/test_sim_runner.py`, `scripts/plot_closed_loop.py`, `tests/test_envelope.py`.
- Type fix in `src/sim/runner.py` fallback: `_pd.DataFrame` -> `pd.DataFrame`.
- Tidy long lines and unused in `scripts/build_coupled_system.py`; add experimental note.
- Tests: `pytest -q` passing.
- Cleanup: removed old run artifacts and caches; added ignores
  - Deleted: `runs/*`, `logs/*`, `results/*`, `configs/`, egg-info and local caches
  - Preserved: `归档/` as historical backup
  - Updated `.gitignore` to ignore `runs/`, `logs/`, `results/`, and tool caches

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

## 实验实现（主要实验.md 对齐）
- 改进 TSO 边界母线自动选择：按负荷功率 Top-3（`src/tso/network.py::_infer_boundary_defaults`）。
- 新增基线快照脚本：`scripts/exp_baseline.py`，运行 `uv run python scripts/exp_baseline.py --tag baseline`，生成 TSO(case118) 与 DSO(CIGRE MV) 的电压/载流指标与合规性。
- 新增耦合装配脚本：`scripts/exp_coupling.py`，自动将 case118 的 3 个边界母线与 3 条 CIGRE MV 馈线通过 110/20 kV 变压器 merge_nets；输出合并网 `coupled_net.json` 与指标。
- 新增装置与控制脚本：`scripts/exp_devices.py`，在每个 DSO 根母线添加 BESS(5MW/10MWh) 与等效电容（用 sgen 表达 2 MVAr），实现离散分接位的贪心调节；输出 `devices_net.json` 与指标。
- 运行验证：
  - 基线：已生成 `runs/baseline/*metrics.json`，合规性字段输出正常。
  - 耦合：已生成 `runs/coupled/coupled_net.json` 与 `coupled.metrics.json`。
  - 装置：已生成 `runs/*/devices.metrics.json`（示例 vm_min≈0.863，vm_max≈1.035）；后续可通过包络与本地 NMPC 改善电压越限。

## Solver Robustness & Controller Refactor
- Added shared Pyomo dict utility and tightened solver status checks (`src/utils/pyomo_utils.py`, `src/opt/pyomo_tso.py`, `src/opt/pyomo_dso.py`).
- Logged TSO/DSO solver failures with safe fallbacks and cleaned controller factory (`src/sim/runner.py`).
- Introduced reusable controller base class to unify envelope handling (`src/nmpc/base.py`, `src/nmpc/controller.py`, `src/nmpc/greedy.py`).
 - Centralized solver selection and time-limit handling (`src/utils/solver.py`), defaulting to Gurobi if available, else Ipopt. Both TSO/DSO wrappers accept `solver='auto'` and generic `time_limit_seconds`.
 - `sim/scenario.py`: wired `solvers.time_limit_seconds` and per-solver options; fallbacks preserved.
 - `app/main.py`: now resolves config names from `config/`, `cfg/`, `configs/`; added `--list-configs`.
- Migrated primary configs to `config/` and updated scripts/docs to reference it exclusively. Archived old `cfg/*.yaml` to `归档/cfg_backup/`.
- Relocated batch runner to utilities and updated callers (`src/utils/batch.py`, `src/coord/admm.py`, `src/coord/__init__.py`, `scripts/run_experiment.py`, `tests/test_batch.py`).
- Synced README run instructions and module references with current layout (`README.md`).

- [x] LV three-snapshot scenarios (on_peak_566/off_peak_1/off_peak_1440) with KPIs → runs/lv_snaps/*.metrics.json, lv_snapshots.summary.json
  - command: `uv run python scripts/exp_lv_snapshots.py --tag lv_snaps`
- [x] 24h timeseries (no coordination) using DFData/ConstControl, logs voltages/loading/PCC/BESS and SoC → runs/ts_local/*
  - command: `uv run python scripts/exp_timeseries_local.py --tag ts_local --dt-min 5`
- [x] TI-NMPC wrapper hooked to scenario configs (OUR/B3) → runs/smoke_test_our
  - command: `uv run python scripts/exp_tinmpc.py --cfg config/smoke_test_our.yaml --tag our_demo`

- [x] KPI aggregator added: `scripts/aggregate_kpis.py` → writes `report.json/csv`
- [x] 24h TI configs prepared: `config/ti24h_our.yaml`, `config/ti24h_b3.yaml`
  - run: `uv run python scripts/exp_tinmpc.py --cfg config/ti24h_our.yaml --tag ti24h_our`
  - run: `uv run python scripts/exp_tinmpc.py --cfg config/ti24h_b3.yaml --tag ti24h_b3`
- [x] Run index writer: `scripts/make_run_index.py` to generate `INDEX.md` per run
