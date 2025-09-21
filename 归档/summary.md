# 项目代码审查摘要与改进建议

本文档对整个项目代码进行了全面审查，并总结了潜在的错误、设计问题和改进建议，按优先级排序。

---

## 1. 潜在风险与 Bug (高优先级)

### 1.1. 优化求解器失败时静默处理

- **文件**: `src/sim/runner.py` (内部 `tso_solver`, `dso_solver` 函数), `src/opt/pyomo_tso.py`, `src/opt/pyomo_dso.py`
- **问题**: 当 Pyomo 求解器因任何原因（如模型无解、收敛失败）失败时，程序会捕获一个通用的 `Exception`，然后返回一个全零的向量，并且**不会记录任何错误日志**。
- **风险**: 这种“静默失败”是**非常危险的**。它会隐藏底层的严重问题，导致仿真在错误数据的基础上继续运行，最终产出具有严重误导性的结果，而表面上却没有任何异常。
- **建议**:
    1.  在 `runner.py` 的 `except` 块中，增加明确的错误日志记录，例如 `logging.error("TSO solver failed: %s", exc)`。
    2.  在 `opt` 目录的 `solve_*_model` 函数中，对求解器的返回状态做更严格的检查。如果状态不是 `optimal` 或 `feasible`，应该抛出异常或至少记录一个严重的警告，而不是让程序继续。

```python
# src/sim/runner.py -> tso_solver

# 问题代码
except Exception as exc:
    # Solver unavailable or failed: produce zero vector fallback
    flows_vec = np.zeros(len(boundary), dtype=float)
    obj = None
    theta_meta = {}

# 建议修改
except Exception as exc:
    # 记录错误日志，以便追踪问题
    import logging
    logging.error("TSO solver failed with exception: %s", exc, exc_info=True)
    # 依旧可以返回零向量，但问题已被暴露
    flows_vec = np.zeros(len(boundary), dtype=float)
    obj = None
    theta_meta = {}
```

---

## 2. 设计与结构问题 (中优先级)

### 2.1. `run_batch` 函数位置不当

- **文件**: `src/coord/admm.py`
- **问题**: `run_batch` 是一个通用的批处理工具函数，其功能是“根据不同种子多次运行仿真”。它与 ADMM 算法本身没有直接关系。将其放在 `coord.admm` 模块中，违反了“高内聚、低耦合”的设计原则，使代码结构混乱。
- **建议**: 将 `run_batch` 函数移动到一个更通用的模块，例如 `src/utils/batch.py` 或 `src/sim/batch.py`。

### 2.2. 算法选择逻辑僵化

- **文件**: `src/sim/runner.py`
- **问题**: 代码中使用 `if alg == "B0": ...` 和 `if alg == "B1": ...` 的硬编码来分发不同的控制策略。当未来需要增加更多算法（如 B2, B4, ...）时，这个 `if/elif/else` 链会变得越来越长，难以维护。
- **建议**: 应用**策略设计模式 (Strategy Pattern)**。
    1.  定义一个所有控制器都遵循的 `BaseController` 接口（例如，都有 `run_step` 方法）。
    2.  将 `NMPCController`, `GreedyController`, `B1Controller` 等实现为该接口的不同策略类。
    3.  在 `_build_controller` 函数中，根据配置字符串 (`alg`)，使用一个工厂来实例化正确的控制器策略对象。

### 2.3. 控制器代码重复

- **文件**: `src/nmpc/controller.py`, `src/nmpc/greedy.py`
- **问题**: `NMPCController` 和 `GreedyController` 的 `__init__` 方法中，有完全相同的 `self.envelope` 初始化代码。
- **建议**: 结合上一点，可以创建一个 `BaseController` 基类，将这部分共享的初始化逻辑放在基类的 `__init__` 方法中，然后让具体的控制器类继承它。

---

## 3. 代码冗余与坏味道 (低优先级)

### 3.1. Pyomo 模型构建代码冗余

- **文件**: `src/opt/pyomo_dso.py`, `src/opt/pyomo_tso.py`
- **问题**: 这两个文件中存在大量重复的模板代码，用于将 `numpy` 数组或 `pandas` 数据转换为 Pyomo `Param` 所需的 `dict` 格式。
- **建议**: 创建一个通用的工具函数来处理这种转换，以减少重复代码。

```python
# 建议在 src/utils/pyomo_utils.py 中新增
import numpy as np

def ndarray_to_pyomo_dict(arr: np.ndarray) -> dict:
    """Converts a numpy array to a Pyomo-compatible dict."""
    if arr.ndim == 1:
        return {i: float(v) for i, v in enumerate(arr)}
    elif arr.ndim == 2:
        return {(i, j): float(arr[i, j]) for i in range(arr.shape[0]) for j in range(arr.shape[1])}
    raise ValueError("Only 1D and 2D arrays are supported")

# 然后在 pyomo_tso.py 中使用
# model.Y = pyo.Param(model.B, model.B, initialize=Y_dict, mutable=False)
# 可以简化为
# model.Y = pyo.Param(model.B, model.B, initialize=ndarray_to_pyomo_dict(Y), mutable=False)
```

### 3.2. 重复的 Pandas 导入

- **文件**: `src/sim/runner.py`
- **问题**: 在 `_build_controller` 函数内部的两个不同分支中，都出现了 `import pandas as pd`，而该文件顶部已经有了全局的导入。
- **建议**: 删除函数内所有重复的 `import pandas as pd` 语句。

### 3.3. 不规范的变量存在性检查

- **文件**: `src/sim/runner.py`
- **问题**: 代码中使用了 `if 'dso_indices' in locals() or 'dso_indices' in globals()` 来检查变量是否存在。这种方式不标准、可读性差且脆弱。
- **建议**: 改为使用标准的 `if dso_indices is not None:` 进行检查。

---

## 4. 文档与配置不一致

### 4.1. `README.md` 与项目实际结构脱节

- **文件**: `README.md`
- **问题**: `README.md` 作为项目的设计蓝图，其内容与当前代码库的实际情况存在多处不一致。
    - **路径不一致**: `README` 中提到的 `io/` 和 `ctrl/` 目录在 `src/` 中不存在。
    - **包名不一致**: `README` 中使用的包名是 `tdnmpc`，而 `pyproject.toml` 中定义的是 `codex-tso-dso-nmpc`。
- **建议**:
    - **选项A (推荐)**: 更新 `README.md`，使其与当前的项目结构、文件路径和包名保持一致。
    - **选项B**: 在 `README.md` 的顶部添加一个说明，指出该文件是一个初始设计文档，部分内容可能已过时，实际实现请以 `src` 目录为准。
