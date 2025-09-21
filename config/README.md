Config files centralize all tunable parameters for one-click experiments.

How to choose configs
- CLI: `uv run codex-nmpc --config <name>`
  - Resolves `<name>` as `config/<name>.yaml` (also supports `.yml`/`.json`).
  - `--list-configs` prints available configs under `config/`.

Top-level keys
- seed: int (default 0)
- time: { start: str, steps: int, dt_min: int }
- N_pred: int (defaults to `time.steps`)
- algorithm: OUR | B3 | B1 | GREEDY (default OUR)
- admm: { rho: float, max_iters: int, tol_primal: float, tol_dual: float }
- envelope: { margin: float, alpha: float }
- solvers:
  - tso: gurobi | ipopt (default gurobi)
  - dso: gurobi | ipopt (default follows tso)
  - time_limit_seconds: float (global time cap; mapped per solver)
  - tso_options: dict (passed to solver, e.g., { max_iter: 300 })
  - dso_options: dict
- tso: { admittance: 2D array, injections: 1D array, boundary: 1D array, cost_coeff: float, bounds?: {lower: 1D, upper: 1D}}
- dsos: list of DSO entries with:
  - sens: { Rp: 2D, Rq?: 2D, vm_base: 1D }
  - profiles: { csv?: path, load?: [..], pv?: [..] }
  - vmin: float (default 0.95), vmax: float (default 1.05)
  - penalty_voltage: float (default 1e4), cost_coeff: float (default 50.0)
- forecast: { sigma_load: float, sigma_pv: float, rho: float }
- ti_envelope: { enabled: bool, alpha: float, scenario_count: int, margin: float, penalty?: float }
- mapping: { "<tso_bus>": [dso_index, dso_bus_index], ... }
- run: { base: path, tag?: str }
- logging: { base_dir: path, log_file: str, metrics_file: str, level: str }
 - plots:
   - admm_per_step:
     - enabled: bool (default true)
     - latest_only: bool (default true; overwrite a single admm_conv_latest.png)
     - stride: int (default 0; when >0 and latest_only=false, saves every k steps)
     - dir: str (optional subdirectory under run_dir for step plots)

Defaults
- If `N_pred` missing, use `time.steps`.
- Solver: specify gurobi or ipopt explicitly per run; both TSO/DSO use the same by default.
- Time limits: `time_limit_seconds` → Gurobi `TimeLimit`, Ipopt `max_cpu_time`.
 - Plot outputs: by default only `admm_conv_latest.png` is kept to avoid file explosion; set `plots.admm_per_step.latest_only: false` and a `stride` to save periodic snapshots.

Examples
- See `config/demo.yaml`, `config/smoke_test_our.yaml`, `config/smoke_test_b3.yaml`, `config/ti24h_our.yaml`, `config/ti24h_b3.yaml`.
