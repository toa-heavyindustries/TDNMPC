
# AGENTS.md

面向对象：AI 编码代理（如 Codex CLI/IDE/Cloud）与项目贡献者  
目标：一次跑通、可复现、可持续维护

---


## 1) 目录结构约定（src-layout）

```
.
├── src/                       # 主要代码和模块文件夹
├── tests/                     # 测试（pytest）
├── notebooks/                 # 研究用 Notebook（轻依赖，不写业务逻辑）
├── scripts/                   # 命令脚本（统一用 uv run 执行）
├── data/                      # 小样本/合成数据（大数据请用 DVC/LFS，或外部存储）
├── pyproject.toml             # 依赖与工具配置（PEP 621）
├── uv.lock                    # 锁文件（提交到仓库）
├── .pre-commit-config.yaml    # 质量钩子
└── AGENTS.md                  # 本文件
```

约定：**不提交**大体积/敏感数据；`data/` 仅放小样本或生成脚本。

---

## 2) 依赖与环境（统一用 uv）

### 2.1 添加/移除依赖

```bash
# 生产依赖
uv add "pandas>=2.2" numpy

# 开发依赖（dev 组）
uv add --group dev ruff mypy pytest pre-commit

# 移除依赖
uv remove numpy
```

### 2.2 同步与锁定

```bash
# 标准同步（更新 uv.lock 与 .venv）
uv sync

# 仅装特定组（示例：CI 只装 prod 组）
uv sync --no-group dev --group prod

# 以锁定模式运行（禁止改锁）
uv run --locked pytest -q

# 离线/空管网络
UV_OFFLINE=1 uv sync --locked
```

### 2.3 固定 Python 版本

```bash
uv python pin 3.11
```

### 2.4 兼容无 uv 环境（导出 requirements）

```bash
uv export --format requirements-txt > requirements.txt
```

---

## 3) 统一命令（一律通过 uv 执行）

### 代码质量

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
```

### 测试

```bash
uv run pytest -q
uv run pytest -q -k "not slow"
```

### 提交前自动化

```bash
uvx pre-commit run -a
```

### 文档（如使用 mkdocs/sphinx，按需启用）

```bash
uv run mkdocs build -q
```

> 说明：`uvx <tool>` 可在**临时隔离环境**运行命令行工具（类似 pipx），避免污染项目环境。

---

## 4) 提交 / PR 规范（硬性要求）

* **必须通过**：`ruff check`、`mypy --strict`、`pytest -q`
* **PR 标题**：`[module] 简述变更`
* **PR 描述**：动机→方案→影响面→回归风险→测试点
* 任何公共 API 变更需附测试与文档更新

本地与 CI 的最小校验脚本：

```bash
uv run ruff check . && uv run mypy --strict src && uv run pytest -q
```

---

## 5) 复现实验要求（科研项目）

* 固定随机种子；记录 `uv.lock` 与 `.python-version`
* 产出至少 `metrics.json` 与图表 `figures/`
* 大数据/模型产物用外部存储或 DVC/LFS；仓库保留生成脚本

---

## 6) 代理工作流规则（给 AI Coding Agent）

### 6.1 当我说“跑测试/检查”

```bash
uv run ruff check . && uv run mypy --strict src && uv run pytest -q
```

### 6.2 当我说“修格式”

```bash
uv run ruff format .
uvx pre-commit run -a
```

### 6.3 当我说“加依赖 X 到 dev 组”

```bash
uv add --group dev X
uv sync
```

### 6.4 当我说“导出 requirements”

```bash
uv export --format requirements-txt > requirements.txt
```

### 6.5 危险/需审批的操作（**不要自行执行**）

* 修改 Shell 启动文件：`~/.zshrc` / `~/.bashrc` / `~/.config/*`
* 写入用户主目录或系统目录、安装系统级包、开启公网访问服务
* 删除非工作区文件、批量格式化超出本仓库的路径
* 任意联网下载/上传敏感数据

---

## 7) 故障排查

```bash
# 查看可用 Python / 校正版本
uv python list

# 清理/瘦身缓存
uv cache prune

# 查看当前环境包
uv pip list
uv pip freeze
```

若遇到 “命令找不到/包不可见”，不要 `pip install`，统一改用：

```bash
uv add <pkg>        # 写入 pyproject.toml 并更新 uv.lock
uv sync             # 同步环境
```

---

## 8) Monorepo 约定（如适用）

* 根目录放一份通用 `AGENTS.md`；子包可放更细的 `apps/*/AGENTS.md` 或 `packages/*/AGENTS.md`
* 代理在子包范围内工作时，以**最近**的 `AGENTS.md` 为准
* 所有子包共用根级 `pyproject.toml/uv.lock`（除非另有声明）

---

## 9）进行较长任务时提前说明任务计划，收到回复OK后再继续


## 10）编码风格
1. 代码可读性要好
2. 变量名称可读性要好，在长度和可读性之间做好平衡
3. 在满足可读性的基础上尽量简洁
4. 如非必要，尽量避免使用继承的设计模式
5. 尽量避免硬编码
6. 每个函数实现的功能不应过于复杂
7. 在可能的情况下模块化
8. 便于其他项目复用
9. 相对复杂的功能如果有库能够实现应该优先用库而非自己实现
10. 避免使用过于冷门或者可读性差的写法，程序应该直观、简洁
11. 程序设计要有全盘规划意识
12. 如要修改参数，应该使用配置文件的形式

## 11）每完成一个模块，就在progress.md中添加相关记录
