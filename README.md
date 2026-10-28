# Trader

## 项目简介（Overview）

**Trader** 是基于 **LangGraph** 构建的多智能体金融市场研究框架，基于 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 进行 CLI 化与本地化适配。它让不同角色的 Agent 分工完成市场研究、多空辩论、交易计划与风险评估，最终生成研究结论和结构化报告。

**输入**：标的代码（Ticker）、分析日期、可选的持仓信息与分析师组合。  
**输出**：最终决策与评级、多阶段 Agent 研究报告（Markdown / HTML）及运行记录。

项目以**研究分析与辅助决策**为目标，不执行真实账户交易。

## 核心功能（Features）

- **多 Agent 协同研究**：市场、基本面、新闻、情绪分析师分工研究；多空研究员辩论，随后由交易员、风险团队和投资组合经理形成最终结论。
- **LangGraph 工作流**：支持分析师并行执行、独立工具调用子图、共享状态传递以及研究和风险阶段的条件路由。
- **多市场数据接入**：通过统一数据路由使用行情、技术指标、财报和新闻数据；**A 股已完成真实 LLM 全流程验收**，可按股票代码自动选择国内数据源。
- **统一 LLM 配置**：使用 `LLM_API_KEY`、`LLM_MODEL` 和 `LLM_BASE_URL` 接入 OpenAI-compatible API；底层保留多模型提供商客户端。
- **交互式命令行**：通过 `trader` 选择标的、日期、分析师和研究深度，也支持参数化、非交互运行。
- **报告与执行恢复**：输出分阶段 Markdown 报告、完整 HTML 报告及日志；支持历史决策记忆和可选的检查点恢复。
- **历史回测与持仓分析**：支持多个标的/日期的历史决策评估，可传入持仓文件供交易和风险 Agent 参考。

## 项目架构（Architecture）

### Agent 工作流程

```text
输入：Ticker / 日期 / 可选持仓
                |
        +-------+-------+------------------+
        |       |       |                  |
      Market   News  Fundamentals       Sentiment
      Analyst Analyst   Analyst          Analyst
        |       |       |                  |
        +-------+-------+------------------+
                |  （选定的分析师并行运行）
                v
          Bull / Bear Researchers
              多空辩论
                |
          Research Manager
                |
              Trader
                |
     Aggressive / Neutral / Conservative
              风险讨论
                |
          Portfolio Manager
                |
       最终决策、评级与研究报告

Memory Log：与分析师阶段并行处理历史决策，
            为后续组合决策提供历史经验。
```

实际流程由 LangGraph 管理：分析师在独立子图中调用数据工具，研究辩论与风险讨论按条件路由推进，最终由 Portfolio Manager 汇总。

### 工程架构

```text
CLI / Backtest / Python API
             |
     TradingAgentsGraph
       (LangGraph Runtime)
             |
   +---------+----------+
   |                    |
 Agents               Graph
 (业务角色)      (状态 / 路由 / 检查点)
   |                    |
   +---------+----------+
             |
   +---------+----------+-----------+
   |                    |           |
LLM Clients          Dataflows     Memory
(模型接入)          (数据源路由)   (历史经验)
   |                    |           |
   +---------+----------+-----------+
             |
       决策 / 报告 / 日志
```

## 快速开始（Quick Start）

**环境要求**：Python 3.11+，Git，以及可访问的 LLM API。以下以 Windows PowerShell 和 Conda 为例。

**1. 克隆与安装**

```powershell
git clone https://github.com/wangd237/Trader.git
cd Trader
conda create -n trader python=3.11 -y
conda activate trader
python -m pip install -e .
Copy-Item .env.example .env
```

如需分析 **A 股**，额外安装数据源依赖：

```powershell
python -m pip install -e ".[ashare]"
```

**2. 配置模型**

编辑项目根目录的 `.env`，填写实际可用的 OpenAI-compatible API 信息。以下为示例：

```dotenv
LLM_API_KEY=your_api_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
TRADINGAGENTS_OUTPUT_LANGUAGE=Chinese
TRADINGAGENTS_RESULTS_DIR=./results
```

`LLM_BASE_URL` 填 API 基础地址，不是 `/chat/completions` 接口；请勿提交包含密钥的 `.env`。

**3. 启动**

```powershell
trader
```

CLI 会引导选择标的、分析日期、分析师和研究深度。运行过程中会展示 Agent 执行进展；完成后可保存研究报告。

## 使用方式（Usage）

**分析美股（非交互）**

```powershell
trader --ticker NVDA --date 2026-10-20 --analysts market,news --save --no-show
```

**分析 A 股**

```powershell
trader --ticker 600519.SH --date 2026-09-30 --analysts market,fundamentals,news,social --save --no-show
```

A 股行情及技术指标、财务报表和公告/新闻通过国内数据源适配；无需修改 Agent 工作流。以上示例运行时会调用 LLM，产生 API 使用费用。

**历史回测**

```powershell
trader backtest NVDA --start 2026-09-01 --end 2026-09-15 --every 7
```

**常用参数**

| 参数 | 用途 |
| --- | --- |
| `--ticker` / `--date` | 指定分析标的与日期（YYYY-MM-DD） |
| `--analysts` | 选择分析师：`market`、`news`、`fundamentals`、`social` |
| `--save` / `--no-show` | 保存报告 / 不在结束时展示完整报告 |
| `--html` / `--no-html` | 控制是否额外保存 HTML 报告 |
| `--portfolio` | 读取包含持仓及现金信息的 JSON 文件 |
| `--checkpoint` | 启用节点级检查点及中断恢复 |

查看完整命令说明：`trader --help`、`trader backtest --help`。

**输出位置**：默认报告位于 `<TRADINGAGENTS_RESULTS_DIR>/reports/<Ticker>_<时间戳>/`，包含 `complete_report.md`、`complete_report.html` 和各 Agent 的分阶段报告。CLI 运行日志另存于 `<TRADINGAGENTS_RESULTS_DIR>/<Ticker>/<分析日期>/`。

## 项目结构（Project Structure）

```text
Trader/
├── cli/                         # CLI 入口、交互引导、实时状态展示
├── tradingagents/
│   ├── agents/                  # 分析师、研究员、交易员及风控等 Agent
│   ├── graph/                   # LangGraph 编排、路由、状态和检查点
│   ├── dataflows/               # 市场数据工具、提供商路由与缓存
│   │   └── vendors/             # Yahoo、A 股等数据源适配
│   ├── llm_clients/             # 模型客户端及提供商工厂
│   ├── memory/                  # 历史决策记录与反思
│   ├── reporting.py             # Markdown 报告生成
│   ├── report_html.py           # HTML 报告生成
│   └── backtest.py              # 历史回测
├── tests/                       # 自动化测试
├── .env.example                 # 环境变量示例
├── pyproject.toml               # 依赖与 CLI 命令注册
└── main.py                      # Python 调用示例
```
