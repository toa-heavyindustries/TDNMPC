## 基本实验实施计划（v1）

> 依据：基本实验.md 与 报告2.md；目标：一次跑通、可复现、可扩展。

- 时间轴：Δt=5min，T=24h（288 点）；默认 N_pred=48，N_ctrl=12。
- 输出根目录：`runs/<tag>/` 与 `results/`（图表、汇总）。
- 统一命令：均通过 `uv run` 执行；配置统一用 JSON/YAML。

## 里程碑概览（M1–M4）

- M1 数据与预测：生成 24h 负荷/PV/温度；支持含相关性的多场景预测。
- M2 配网与灵敏度：IEEE-33（先行）→ 提取 LinDistFlow 灵敏度并校验。
- M3 控制与协调：NMPC + ADMM（既有）扩展算法开关；接入 TI 包络；集中式/分布式/鲁棒基线。
- M4 仿真与评估：滚动仿真、记录与可视化，批量运行与汇总。

## 具体实施步骤（按顺序推进）

1) 初始化与环境
- 校验环境：`uv run ruff check . && uv run pytest -q`
- 若无数据：创建 `data/` 目录；准备 `runs/` 输出目录。

2) M1 数据与预测（src/profiles.py, src/sim/forecast.py, scripts/make_profiles.py）
- 生成 24h 典型日曲线：
  - `uv run python scripts/make_profiles.py --date 2024-01-01 --freq 5min --seed 42 --out data/profiles.csv`
  - 产物：`data/profiles.csv`，图表保存到 `runs/<ts>/profiles.png`
- 新增多场景预测（需开发）：
  - 在 `src/sim/forecast.py` 实现：`sample_forecast(truth: pd.Series, sigma: float, rho: float, horizon: int, n: int) -> np.ndarray`
    - 噪声模型：AR(1)，`e_t = rho*e_{t-1} + σ*ξ_t`，ξ~N(0,1)；返回形状 `(n, horizon)`。
  - 可选：在 `data/scenarios.yaml` 维护误差等级集合（σ_L, σ_PV, ρ）。

3) M2 配网与灵敏度（src/dso/network.py, src/models/lindistflow.py, scripts/build_dso.py）
- 构建 IEEE-33：`uv run python scripts/build_dso.py --case ieee33 --out data/ieee33.json`
- 校验线性化：自动生成 `runs/<ts>/lin_check.json`（包含 mae/max/passed）。
- 在 `src/dso/network.py` 暴露灵敏度接口（若缺）：
  - `get_sensitivity(net, method="lindistflow"|"local", at_state=None) -> dict[str, np.ndarray]`

4) M3 控制与协调（已具雏形，需补完）
- 算法开关与控制器（src/nmpc/controller.py, src/sim/runner.py）
  - 在配置中增加 `algorithm: B0|B1|B2|B3|OUR`；`runner` 根据算法选择控制器/求解流程。
  - B0（贪心基线）：新增 `src/nmpc/greedy.py`，越限再校正。
  - B1（集中式）：在 ADMM 外提供一次性联合求解路径（可先用 TSO/DSO 串联近似）。
  - B2（分布式、无 TI）：使用现有 ADMM + 单场景预测。
  - B3（全场景鲁棒）：扩展 DSO 模型支持多场景硬约束（先期可降维到边界注入约束）。
  - OUR（分布式 + TI）：使用多场景预测构造包络并施加到模型。
- TI 包络（src/coord/ti_env.py 与 Pyomo 接口）
  - 由多场景预测得到时间-接口维度的上下界；
  - 在 DSO/TSO Pyomo 模型中对“边界注入相关变量”施加 `lower/upper` 约束（需在 `opt/pyomo_dso.py` 暴露可控注入或聚合变量，并在 `sim/runner.py` 里映射到接口顺序）。
- 灵敏度在线更新
  - 在 `runner` 的滚动循环内，按 `cfg['sensitivity']['update_every']` 触发重算或缓存更新。

5) M4 仿真与评估（src/sim/runner.py, scripts/run_experiment.py, src/eval/metrics.py, scripts/evaluate_run.py, src/viz/plots.py）
- 配置组织：在 `cfg/` 下新增示例 `*.yaml` 用于 6 个“冒烟”用例。
- 单次运行：`uv run python scripts/run_experiment.py --cfg cfg/our_33.yaml --tag our_33`
- 批量运行：`uv run python scripts/run_experiment.py --cfg cfg/our_33.yaml --seeds 1 2 3 4`
- 评估与作图：
  - `uv run python scripts/evaluate_run.py runs/<tag>` → `metrics.json`
  - 必要图表：电压热图、PCC 轨迹、ADMM 收敛曲线。

## 6 个冒烟用例与执行顺序（先跑通）

1) Sanity-33-Deterministic（B1）
- 配置：`ieee33`，σ=0%，TI 关，集中式 NMPC。
- 目的：验证建模/流程正确性与单位/量纲。

2) LinCheck-33（灵敏度校验）
- 运行：`scripts/build_dso.py`（K=100 样本）；检查 `lin_check.json` 中 mae/max 误差。

3) Our-33-Moderate（OUR）
- 配置：σ_L=5%，σ_PV=15%，ρ=0.6；TI 开；分布式（ADMM K=3）。
- 指标：越限率、弃电量、单步时间、收敛步数。

4) Baseline-B2-33（B2）
- 与用例 3 相同但 TI 关；作横向对比。

5) Our-123-Moderate（OUR，后续）
- 同 3，但 `ieee123`（若暂缺，可先用 33 节点复用配置，预留 TODO）。

6) Our-33-RealtimeTight（OUR）
- 同 3，但 `time_budget_s=1.0`；关注 95% 步是否满足预算。

## 配置模板（示例，YAML）

seed: 20250920

algorithm: OUR  # B0|B1|B2|B3|OUR

time:
  start: "2024-01-01 00:00"
  steps: 288
  dt_min: 5

admm:
  rho: 1.0
  max_iters: 50
  tol_primal: 1.0e-4
  tol_dual: 1.0e-4

solvers:
  tso: ipopt
  dso: ipopt

tso:
  admittance: ...  # from data/tso_case.json or inline
  injections: ...
  boundary: ...
  cost_coeff: 10.0

dsos:
  - sens: {Rp: ..., Rq: ..., vm_base: ...}
    profiles: {csv: data/profiles.csv}
    vmin: 0.95
    vmax: 1.05
    penalty_voltage: 1.0e3
    cost_coeff: 50.0

sensitivity:
  method: lindistflow
  update_every: 6

forecast:
  sigma_load: 0.05
  sigma_pv: 0.15
  rho: 0.6
  scenario_count: 8

ti_envelope:
  enabled: true
  alpha: 0.9

run:
  base: runs
  tag: our_33

mapping:
  "2": [0, 0]  # TSO boundary bus -> (dso_index, dso_bus_index)

## 变更清单（需开发/补全）

- [ ] `src/sim/forecast.py`：新增 `sample_forecast(...)` 多场景预测（AR(1)）。
- [ ] `src/dso/network.py`：补充 `get_sensitivity(...)` 与（可选）`solve_ac(...)` 辅助；暴露 IEEE-123 TODO。
- [ ] `src/opt/pyomo_dso.py`：暴露/映射“边界注入”以便施加 TI 包络上下界；支持多场景扩展（B3）。
- [ ] `src/sim/runner.py`：
  - 支持 `algorithm` 开关（B0/B1/B2/B3/OUR）。
  - 滚动预测接入（调用 `sample_forecast`）；按 `update_every` 更新灵敏度。
  - OUR：在求解前根据多场景构造包络并注入模型约束。
- [ ] `src/nmpc/greedy.py`：实现 B0 贪心基线（越限再校正）。
- [ ] `cfg/*.yaml`：为 6 个用例各给 1 份样例配置（命名与标签一致）。
- [ ] 文档与脚本：在 README/AGENTS 中补充一键命令；脚本打印关键路径/指标。

## 产出与验收

- 每次运行生成：`runs/<tag>/{logs.csv, summary.json, admm_history_step_*.csv, *.png}`；
- 指标：`runs/<tag>/metrics.json`（由 `scripts/evaluate_run.py` 生成）。
- 图表：电压热图、PCC 轨迹、ADMM 收敛图；
- 通过标准：
  - Sanity-33：收敛、无异常越限；
  - LinCheck-33：mae < 3%、max < 6%（可按网型微调）；
  - Our vs B2：越限率与弃电量显著下降；单步时间接近；
  - RealtimeTight：>95% 步满足时间预算（统计日志）。

## 一键复现实验（建议汇总）

- 生成曲线：`uv run python scripts/make_profiles.py --freq 5min --seed 42 --out data/profiles.csv`
- 构建配网+校验：`uv run python scripts/build_dso.py --case ieee33 --out data/ieee33.json`
- 运行用例：`uv run python scripts/run_experiment.py --cfg cfg/our_33.yaml --tag our_33`
- 评估与作图：`uv run python scripts/evaluate_run.py runs/our_33`
