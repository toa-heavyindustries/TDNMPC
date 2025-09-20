# Progress Log

- Step 1 (Profiles) — 完成时间 2025-09-20T10:23:46Z: 搭建 profile 生成模块、CLI、单测，并通过 `uv run pytest -q`。
- Step 2 (DSO + LinDistFlow) — 完成时间 2025-09-20T10:36:39Z: 构建 IEEE33 网模、导出/加载、线性化灵敏度与校验脚本，新增 CLI 与单测通过。
- Step 3 (TSO DC) — 完成时间 2025-09-20T10:39:20Z: 生成 DC 输电案例、标记边界、求解功率平衡，提供 CLI 与单测通过。
- Step 4 (Coupling) — 完成时间 2025-09-20T10:43:35Z: 编排 TSO-DSO 接口映射、信号交换与残差计算，提供示例脚本与单测通过。
- Step 5 (Time Windows) — 完成时间 2025-09-20T10:45:37Z: 实现 Horizon/滚动窗口工具与 profile 对齐函数，并补充单测通过。
- Step 6 (Pyomo DSO) — 完成时间 2025-09-20T10:48:50Z: 搭建基于 LinDistFlow 的 Pyomo 下层调度模型，支持求解与结果抽取并通过单测（若无求解器仅跳过）。
- Step 7 (Pyomo TSO) — 完成时间 2025-09-20T12:12:46Z: 构建上层 DC Pyomo 模型（边界约束/调节变量），提供求解与结果抽取并通过单测（若无求解器仅跳过）。
- Step 8 (ADMM) — 完成时间 2025-09-20T12:16:05Z: 实现 TSO-DSO 之间的共识 ADMM 框架，支持残差历史记录与收敛判定，并通过单测。
- Step 9 (TI Envelope) — 完成时间 2025-09-20T12:20:00Z: 提供包络创建/更新与违约统计工具，完善协调约束监控并通过单测。
- Step 10 (NMPC Wrapper) — 完成时间 2025-09-20T12:24:38Z: 搭建基于 ADMM 的 NMPC 控制器封装，支持包络更新与残差输出，并补充单测通过。
