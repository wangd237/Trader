# Trader

**Trader** 是一个以命令行优先的多智能体金融市场研究框架，基于 TradingAgents 的开源实现进行高保真移植。它将行情、基本面、新闻和市场情绪交给不同的分析 Agent，再经过多空辩论、交易计划和风险评议，形成最终研究结论。

项目当前重心是复现和验证原始工作流，而不是重写底层框架或开发网页应用。生成的 HTML 报告是离线文件，不是 Web 服务。

> **用途声明**：本项目只用于学习、研究和实验。模型可能产生错误分析；输出不构成投资建议，也不应直接用于真实资金的自动交易。

## 工作流程

~~~text
输入：标的代码 + 分析日期 + 可选的持仓信息
                    |
          Market / Fundamentals
          News / Sentiment
          (分析 Agent 并行)
                    |
        Bull / Bear Researchers
             (多空辩论)
                    |
           Research Manager
                    |
               Trader
                    |
     Risk Debators (多方风险审议)
                    |
           Portfolio Manager
                    |
      决策结论 + 分析报告 + 轨迹
~~~

运行层仍沿用参考实现的 LangGraph 编排与工具接口，包含多模型提供商接入、数据源路由、记忆/检查点、报告和历史回测。

## 安装

建议 Python 3.11+；以下以 Windows PowerShell / Conda 为例。

~~~powershell
git clone https://github.com/wangd237/Trader.git
cd Trader
conda create -n trader python=3.13 -y
conda activate trader
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
~~~

打开 .env，根据所使用的模型提供商填入密钥，并选择相应的 Provider 与模型。比如，使用 OpenAI API 时，至少需要设置 OPENAI_API_KEY；也可以改用参考框架支持的其他提供商。**请勿把真实密钥提交到 Git。**

**兼容性约定**：为了不批量改动核心源码，本阶段保留 Python 内部包名 `tradingagents` 和 `TRADINGAGENTS_*` 环境变量前缀；对外项目名称、安装包名称、主 CLI 命令均为 **Trader / trader**。这只是移植兼容性措施，不代表需要安装另一个同名项目。

## 使用

~~~powershell
# 交互式引导选择标的、分析日期、分析师与模型
trader

# 查看可用参数
trader --help

# 已在 .env 填入运行参数后，可使用无交互选项
trader --ticker NVDA --date 2026-10-20 --analysts market,news --save --no-show

# 可选：回测 (会消耗模型调用额度)
trader backtest NVDA --start 2026-09-01 --end 2026-09-15 --every 7
~~~

无交互执行除上述参数外，还需配置模型提供商、模型、输出语言和辩论轮数等；缺少必要项时 CLI 会提示。原始 Python 入口 `python main.py` 也保留，用于演示直接调用 Graph 类。

运行结果默认写入用户目录下的 `~/.tradingagents/logs`，行情缓存和决策记忆位于同级目录；具体位置可用 .env 中的相关设置覆盖。报告包括 Markdown，以及可选的独立 HTML 页面；无需网页服务。

## 代码结构

| 路径 | 责任 |
| --- | --- |
| `tradingagents/agents/` | 各类分析师、研究员、交易员、风险及组合管理 Agent |
| `tradingagents/graph/` | LangGraph 工作流编排、状态传递、条件路由及检查点 |
| `tradingagents/dataflows/` | 市场数据获取、供应商路由与缓存 |
| `tradingagents/llm_clients/` | 多种模型 Provider 的统一接入 |
| `tradingagents/memory/` | 决策历史与经验回顾 |
| `tradingagents/backtest.py` | 历史日期网格回测 |
| `tradingagents/reporting.py` | Markdown 与 HTML 报告 |
| `cli/` | 命令行交互、动态状态展示及参数处理 |
| `tests/` | 参考实现的回归测试及必要的 Trader 适配测试 |

本项目暂不计划单独开发 Web UI、账户登录系统或实盘交易执行服务。

## 开发与测试

~~~powershell
python -m pytest -q
python -m ruff check .
~~~

GitHub Actions 会在提交时执行多版本 Python 测试、干净安装检查与 lint。**源码已迁入不等于真实模型链路已验收**；真实 LLM 与第三方数据 API 的端到端运行需要有效凭证和网络环境。

## 移植进展

| 事项 | 状态 | 备注 |
| --- | --- | --- |
| 导入源码、CLI、配置、测试和许可证 | 完成 | 固定上游提交，保留核心逻辑 |
| 统一对外名称 | 完成 | 包名和旧环境变量暂为兼容而保留 |
| README / 变更记录重新编写 | 完成 | 不使用参考项目 README 文案和图片 |
| 原项目图片资源 | 不移植 | HTML 报告改用小型本地 SVG 占位图 |
| 旧项目公告与发布流程 | 已隔离 | 默认不请求参考项目的公告接口 |
| CI / 回归测试 | 验证中 | 以 Actions 最新运行结果为准 |
| 真实模型 + 数据源 E2E | 未验证 | 需配置密钥后进行实际运行 |
| 复杂网页端 | 不在当前范围 | 以 CLI 和报告输出为主 |

后续所有功能增量都应在对应提交中同步更新本进展表，不提前把未验收能力写成已完成。

## 来源、版本与许可

- **参考代码仓库**：[wangd237/TradingAgents](https://github.com/wangd237/TradingAgents)；固定移植基线提交：[e8f5503](https://github.com/wangd237/TradingAgents/commit/e8f5503e9b86d5c2710b0c76b1f686703968c9df)。
- **上游原始项目**：[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)。
- 当前保留上游 Python API 和内部模块划分，方便最小差异迁移和测试对齐；Trader 后续改动另见 [CHANGELOG.md](CHANGELOG.md)。
- 继承和移植部分遵循 Apache License 2.0，详见 [LICENSE](LICENSE)。保留应当保留的版权与出处信息。

没有沿用上游 README 内容或品牌图片；Trader 的中文文档、占位图及对外命名均为独立整理。
