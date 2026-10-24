# Trader 更新记录

## 初始移植 — 2026-10-24

- 从 wangd237/TradingAgents 提取固定基线 e8f5503e9b86d5c2710b0c76b1f686703968c9df 的 Python 源码、CLI、测试、CI 和许可证。
- 未复制参考项目 README 和展示图片；新增 Trader 专用中文 README 和轻量 SVG 标志占位资源。
- 调整项目包分发名、命令行名称、终端欢迎信息以及 HTML 报告品牌信息；保留内部包名 tradingagents 兼容原代码。
- 默认隔离上游公告检查；不复制原项目的发布工作流和 GitHub Issue 模板。
- 后续进度应同步更新 README。测试通过与真实环境验证状态以实际 CI 与运行记录为准。
