

---


# 全局约定（贯穿所有步骤）

* 全量 type hints、numpy/pandas 为主，日志用 `logging`，随机种子可配置。
* I/O 统一走 `utils/config.py`（加载 YAML/JSON），输出统一放到 `runs/<tag>/...`。
* 每个公共函数写 docstring（含输入/输出/副作用），并配一个最小 pytest。

---
程序接口仅供参考

## 运行场景（CLI）

- 提供统一入口：`codex-nmpc`
- 用法：
  - `uv run codex-nmpc --config cfg/smoke_test_b1.yaml`
  - 读取 YAML/JSON 配置，执行仿真，输出写入 `runs/<tag>/` 与 `logs/<tag>/`。
- 运行结果：
  - `runs/<tag>/logs.csv`：每步 TSO/DSO 向量、残差与 envelope
  - `runs/<tag>/figs/`：默认绘制第 0 接口的 TSO/DSO 轨迹
  - `runs/<tag>/admm_history_step_*.csv/png` 与 `admm_history_all.csv/png`
  - `logs/<tag>/metrics.csv`：精简指标

## 模块结构（更新）

- `sim.runner`：仅负责编排步骤与写入产物。
- `sim.scenario`：构建场景与控制器（toy 与集成模式）。
- `sim.io_utils`：日志初始化、历史/指标/概要与图表写入。

## Step 1｜M1：时序曲线（Profiles）

**目标**：生成/加载 24h 负荷、PV、温度曲线，保存 CSV 并可视化。
**文件**：`profiles.py`, `scripts/make_profiles.py`
**函数**

```python
def make_time_index(date: str, freq: str) -> pd.DatetimeIndex
def gen_load_profile(idx: pd.DatetimeIndex, seed: int = 42) -> pd.Series
def gen_pv_profile(idx: pd.DatetimeIndex, peak_kw: float = 1000.0, seed: int = 42) -> pd.Series
def gen_temp_profile(idx: pd.DatetimeIndex, seed: int = 42) -> pd.Series
def save_profiles(path: Path, **series: pd.Series) -> None
def load_profiles(path: Path) -> dict[str, pd.Series]
def plot_profiles(series: dict[str, pd.Series], out: Path | None = None) -> None
```

**调用**：`scripts/make_profiles.py` 调 `make_time_index→gen_*→save→plot`
**运行**：`uv run python -m codex_tso_dso_nmpc.scripts.make_profiles --out data/profiles.csv`
**期望输出**：`data/profiles.csv`、`runs/.../profiles.png`
**验收**：列齐全（time, load, pv, temp），缺失值=0。

---

## Step 2｜M2：单个 DSO 网模 + LinDistFlow 灵敏度

**目标**：用 pandapower 搭 IEEE-33/123，提取/校准 LinDistFlow 灵敏度。
**文件**：`dso/network.py`, `models/lindistflow.py`, `scripts/build_dso.py`
**函数**

```python
# dso/network.py
def build_ieee33() -> "pp.pandapowerNet"
def ac_power_flow(net) -> pd.DataFrame  # 返回节点电压/支路潮流
def export_net(net, path: Path) -> None
def load_net(path: Path)

# models/lindistflow.py
def linearize_lindistflow(net, base: pd.DataFrame) -> dict[str, np.ndarray]
def predict_voltage_delta(sens: dict, dp: np.ndarray, dq: np.ndarray) -> np.ndarray
def validate_linearization(net, sens, n_samples: int = 20, tol: float = 0.03) -> dict
```

**调用**：`build_ieee33→ac_power_flow→linearize→validate`
**运行**：`uv run python -m codex_tso_dso_nmpc.scripts.build_dso --case ieee33 --out data/ieee33.pkl`
**期望输出**：`data/ieee33.pkl`、`runs/.../lin_check.json`（误差统计）
**验收**：平均电压误差 < 3%，最大 < 5%。

---

## Step 3｜TSO 网模（DC）

**目标**：上层输电 DC 模型骨架与接口母线标注。
**文件**：`tso/network.py`, `models/tso_dc.py`, `scripts/build_tso.py`
**函数**

```python
def build_tso_case(n_bus: int = 30) -> dict
def dc_power_flow(case: dict, injections: np.ndarray) -> dict
def mark_boundary_buses(case: dict, bus_ids: list[int]) -> dict
```

**调用**：`build_tso_case→mark_boundary_buses→dc_power_flow`
**运行**：`uv run python -m codex_tso_dso_nmpc.scripts.build_tso --n-bus 30`
**期望输出**：DC 潮流可运行，返回边界母线表。
**验收**：功率平衡残差 < 1e-6。

---

## Step 4｜输配耦合接口

**目标**：定义 TSO-DSO 边界功率与一致性约束。
**文件**：`coord/interface.py`（新），`utils/config.py`
**函数**

```python
def define_coupling(tso_case: dict, dso_nets: list, mapping: dict) -> dict
def push_tso_signals_to_dsos(signals: np.ndarray, dso_nets: list) -> None
def aggregate_dsos_to_tso(dsos: list) -> np.ndarray
def coupling_residuals(tso_p: np.ndarray, dso_p: np.ndarray) -> dict[str, float]
```

**运行**：小脚本构造假信号计算残差。
**期望输出**：`residual_norm` 数值打印。
**验收**：接口维度一致，残差计算正确。

---

## Step 5｜时间轴与滚动窗口

**目标**：NMPC 时域与滚动窗口工具。
**文件**：`utils/timewin.py`
**函数**

```python
@dataclass
class Horizon: start: pd.Timestamp; steps: int; dt_min: int

def make_horizon(start: str, steps: int, dt_min: int) -> Horizon
def rolling_windows(idx: pd.DatetimeIndex, steps: int) -> list[pd.DatetimeIndex]
def align_profiles(idx: pd.DatetimeIndex, profiles: dict[str, pd.Series]) -> pd.DataFrame
```

**验收**：窗口切片覆盖完整 24h，边界不重叠。

---

## Step 6｜DSO 下层优化（Pyomo）

**目标**：在灵敏度约束下做 DSO 经济调度 + 电压约束。
**文件**：`opt/pyomo_dso.py`
**函数**

```python
def build_dso_model(sens: dict, profiles: pd.DataFrame, horizon: Horizon) -> pyo.ConcreteModel
def add_resources(m: pyo.ConcreteModel, der_caps: dict) -> None  # PV, ESS, 可调负荷
def solve_dso(m: pyo.ConcreteModel, solver: str = "ipopt") -> dict
```

**运行**：构造假 DER 配置，求解一次。
**期望输出**：目标值、功率计划、节点电压预测。
**验收**：电压上下限满足（含少量 slack 可控）。

---

## Step 7｜TSO 上层优化（Pyomo）

**目标**：上层目标（如发电成本/拥塞惩罚）与 DC 约束。
**文件**：`opt/pyomo_tso.py`
**函数**

```python
def build_tso_model(case: dict, demand_forecast: np.ndarray, horizon: Horizon) -> pyo.ConcreteModel
def set_boundary_vars(m: pyo.ConcreteModel, n_interfaces: int) -> None
def solve_tso(m: pyo.ConcreteModel, solver: str = "gurobi|cbc") -> dict
```

**验收**：功率平衡、线流约束可满足，返回边界注入序列。

---

## Step 8｜分布式协调（ADMM/ALADIN 简化版）

**目标**：以边界功率为一致变量的分布式协调。
**文件**：`coord/admm.py`, `scripts/run_admm.py`
**函数**

```python
@dataclass
class ADMMState: z: np.ndarray; u: np.ndarray; rho: float

def admm_init(n_if: int, T: int, rho: float = 1.0) -> ADMMState
def admm_step(state: ADMMState, tso_solve, dso_solve) -> tuple[ADMMState, dict]
def admm_converged(history: list[dict], atol: float = 1e-3) -> bool
```

**调用**：闭环 `tso_solve` / `dso_solve`，更新 `z,u`。
**运行**：`uv run python -m codex_tso_dso_nmpc.scripts.run_admm --T 12`
**期望输出**：`primal/dual residual` 随迭代下降曲线。
**验收**：残差 < 1e-3（或 50 步内单调下降）。

---

## Step 9｜Trajectory-Independent（TI）包络约束

**目标**：构建参考无关的安全包络并随时间滚动收紧。
**文件**：`coord/ti_env.py`
**函数**

```python
def compute_envelope(history: pd.DataFrame, bands: dict[str, float]) -> dict[str, np.ndarray]
def apply_envelope_to_model(m: pyo.ConcreteModel, env: dict, vars_map: dict) -> None
def update_envelope(env: dict, new_obs: dict) -> dict
```

**运行**：用模拟历史构造并施加于 DSO/TSO 模型。
**期望输出**：约束计数、被激活的包络段统计。
**验收**：可控性不被过度收紧（求解成功率 > 95%）。

---

## Step 10｜NMPC 控制器封装

**目标**：把预测、优化、施控打成一步（单 DSO 或耦合后）。
**文件**：`nmpc/controller.py`
**函数**

```python
@dataclass
class NMPCResult: x: dict; u: dict; obj: float; status: str

def nmpc_setup(plant, model_build_fn, horizon: Horizon, cfg: dict) -> dict
def nmpc_step(ctx: dict, measurements: dict) -> NMPCResult
def apply_controls(plant, u: dict) -> None
```

**运行**：假“工况”循环 10 步，打印 obj/status。
**期望输出**：每步求解状态、控制量、约束违背=0 或极小。
**验收**：10/10 步求解成功。

---

## Step 11｜预测模块（简单基线）

**目标**：给负荷/PV/价格提供可替换的基线预测（持平/移动平均）。
**文件**：`sim/forecast.py`
**函数**

```python
def forecast_naive(series: pd.Series, horizon: int) -> np.ndarray
def forecast_sma(series: pd.Series, horizon: int, w: int = 6) -> np.ndarray
```

**验收**：预测长度正确，NAN=0，MAPE 打印用于参考。

---

## Step 12｜仿真编排器

**目标**：把时序、预测、NMPC、网络、协调串起来跑一套场景。
**文件**：`sim/runner.py`, `scripts/run_experiment.py`, `cfg/demo.yaml`
**函数**

```python
def simulate_scenario(cfg_path: Path) -> dict[str, Any]
def simulate_step(state: dict, t: int) -> dict
```

**运行**：`uv run python -m codex_tso_dso_nmpc.scripts.run_experiment --cfg cfg/demo.yaml`
**期望输出**：`runs/<tag>/logs.csv`、中间控制量、接口残差。
**验收**：整套无异常，关键指标生成。

---

## Step 13｜评估指标与可视化

**目标**：统一计算指标与画图。
**文件**：`eval/metrics.py`, `viz/plots.py`
**函数**

```python
# metrics
def voltage_violation(df: pd.DataFrame, vmin=0.95, vmax=1.05) -> float
def energy_cost(costs: pd.Series) -> float
def coupling_rmse(residuals: pd.Series) -> float

# plots
def plot_timeseries(df: pd.DataFrame, cols: list[str], out: Path) -> None
def plot_convergence(hist: pd.DataFrame, out: Path) -> None
```

**验收**：生成 `summary.json`（cost, vv\_rate, rmse）。

---

## Step 14｜多 DSO 扩展与批处理

**目标**：支持 N 个 DSO + 一个 TSO，批量跑不同随机种子/场景。
**文件**：`coord/admm.py`（多从属）、`utils/batch.py`（批处理入口）、`scripts/run_experiment.py`（增加 `--seeds`）
**函数**

```python
def make_multi_dso(n: int, base_targets: np.ndarray) -> list[np.ndarray]
def run_batch(simulator: Callable[[int], dict[str, Any]], seeds: Sequence[int], run_dir: Path) -> pd.DataFrame
```

**验收**：批量完成，生成 `runs/<tag>/batch.csv`。

---

## Step 15｜测试覆盖

**目标**：为关键函数补单测与小型集成测。
**文件**：`tests/`
**建议覆盖**

* `test_profiles.py`：时间索引/保存加载/无缺失
* `test_lindistflow.py`：误差门限
* `test_pyomo_dso.py`：可解且电压不违规
* `test_admm.py`：残差单调下降
* `test_ti_env.py`：包络更新正确
  **验收**：`pytest -q` 全绿（或 xfail ≤ 1）。

---

# codex cli 执行顺序（清单式）

1. Step 0：初始化工程与日志 -> 通过
2. Step 1：`make_profiles.py` -> 生成 CSV/PNG
3. Step 2：`build_dso.py` -> 生成网模与灵敏度校验
4. Step 3：`build_tso.py` -> DC 仿真
5. Step 4：接口模块 -> 打印残差
6. Step 5：时间窗口 -> 单测
7. Step 6：下层 Pyomo -> 求解一次
8. Step 7：上层 Pyomo -> 求解一次
9. Step 8：ADMM -> 收敛曲线
10. Step 9：TI 包络 -> 约束统计
11. Step 10：NMPC 封装 -> 10 步滚动
12. Step 11：预测 -> 基线检查
13. Step 12：`run_experiment.py` -> 全链路
14. Step 13：评估/可视化 -> summary 与图
15. Step 14：多 DSO/批处理 -> batch.csv
16. Step 15：补测试 -> 全绿

---

# 期望日志示例（你能对照快速自检）

```
[profiles] saved to data/profiles.csv shape=(288,3)
[dso] lin_distflow mae=0.012 pu, max=0.031 pu ✓
[tso] dc powerflow balance=3.2e-9 ✓
[coupling] residual_norm: 0.0087 → 0.0013 → 5.6e-4 ✓
[admm] iter=25, r=8.1e-4, s=4.3e-4, obj=1.23e5 ✓
[nmpc] t=07:00Z solve=ok obj=1.21e3 vio=0
[eval] cost=..., vv_rate=0.2%, rmse=3.4e-3 ✓
```

---

## 需要的第三方包（一次性装好）

`numpy, pandas, scipy, pyomo, pandapower, matplotlib, pyyaml, tqdm, pytest`

---

如果你愿意，我可以按这个顺序开始为每个 Step 生成骨架代码和最小脚本（保持可运行 + 最小单测），你只需告诉我从哪个 Step 开始。
