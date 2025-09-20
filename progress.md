# Progress Log

- Step 1 (Profiles) — 完成时间 2025-09-20T10:23:46Z: 搭建 profile 生成模块、CLI、单测，并通过 `uv run pytest -q`。
- Step 2 (DSO + LinDistFlow) — 完成时间 2025-09-20T10:36:39Z: 构建 IEEE33 网模、导出/加载、线性化灵敏度与校验脚本，新增 CLI 与单测通过。
- Step 3 (TSO DC) — 完成时间 2025-09-20T10:39:20Z: 生成 DC 输电案例、标记边界、求解功率平衡，提供 CLI 与单测通过。
- Step 4 (Coupling) — 完成时间 2025-09-20T10:43:35Z: 编排 TSO-DSO 接口映射、信号交换与残差计算，提供示例脚本与单测通过。
- Step 5 (Time Windows) — 完成时间 2025-09-20T10:45:37Z: 实现 Horizon/滚动窗口工具与 profile 对齐函数，并补充单测通过。
