# Trader

## 项目简介

**Trader** 是基于 **LangGraph** 构建的多智能体金融市场研究框架，创建不同角色的 Agent 分工完成市场研究、多空辩论、交易计划与风险评估，最终生成研究结论和结构化报告。

**输入**：标的代码（Ticker）、分析日期、可选的持仓信息与分析师组合。  
**输出**：最终决策与评级、多阶段 Agent 研究报告（Markdown / HTML）及运行记录。

项目以**研究分析与辅助决策**为目标，不执行真实账户交易。

## 核心功能

- **多 Agent 协同研究**：市场、基本面、新闻、情绪分析师分工研究；多空研究员辩论，随后由交易员、风险团队和投资组合经理形成最终结论。
- **LangGraph 工作流**：支持分析师并行执行、独立工具调用子图、共享状态传递以及研究和风险阶段的条件路由。
- **多市场数据接入**：通过统一数据路由使用行情、技术指标、财报和新闻数据；**A 股已完成真实 LLM 全流程验收**，可按股票代码自动选择国内数据源。
- **统一 LLM 配置**：使用 `LLM_API_KEY`、`LLM_MODEL` 和 `LLM_BASE_URL` 接入 OpenAI-compatible API；底层保留多模型提供商客户端。
- **交互式命令行**：通过 `trader` 选择标的、日期、分析师和研究深度，也支持参数化、非交互运行。
- **报告与执行恢复**：输出分阶段 Markdown 报告、完整 HTML 报告及日志；支持历史决策记忆和可选的检查点恢复。
- **历史回测与持仓分析**：支持多个标的/日期的历史决策评估，可传入持仓文件供交易和风险 Agent 参考。

## 项目架构

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

## 快速开始

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

**国内财经新闻优先（默认）**

个股新闻与全球快讯按 `cn_news,yfinance` 顺序尝试。国内源使用 AKShare 接入的**财联社、东方财富和新浪财经**最新快讯，可通过 `python -m pip install -e ".[ashare]"` 安装可选依赖。若要获取**历史财经新闻**，可额外安装 `python -m pip install tushare`，并在 `.env` 设置开通了独立新闻资讯权限的 `TUSHARE_TOKEN`；代码优先按分析时间窗口查询**第一财经、财联社、东方财富和新浪财经**的历史新闻。未开通权限时，不会将最新快讯伪装成过去的新闻。

国内快讯接口通常只覆盖最近资讯。若指定日期窗口没有可核验的新闻，路由器会回退 Yahoo；不会使用分析日期之后的新闻充当历史证据。A 股的公司公告仍按原规则优先走巨潮资讯；美股行情、财报数据源不受这项新闻默认值调整影响。

**A 股与港股国内行情（默认自动识别）**

- **A 股**：例如 `600519.SH`、`000001.SZ`。保留已有腾讯财经历史行情、技术指标本地计算、巨潮资讯公告和新浪财经财报。以证券代码后缀自动路由，不需要调整 `data_vendors`。
- **港股**：例如 `9988.HK`、`0700.HK`。通过 AKShare 的新浪财经 `stock_hk_daily` 获取不复权港元日线（`9988.HK` 会映射为新浪的五位代码 `09988`）；`get_stock_data`、`get_indicators` 和 `get_verified_market_snapshot` 使用同一批截至分析日期的 OHLCV。历史时点请求会过滤之后的价格；过期/无效数据会报告不可用。
- **美股**：恢复原本的默认 `yfinance`；原有 `eastmoney_us` 适配器可由明确的工具/提供商配置选择，但不作为默认源。

安装可选的国内数据依赖：`python -m pip install -e ".[ashare]"`。

单独验证港股数据（不调用 LLM）：

```powershell
python -c "from tradingagents.dataflows.vendors.hk import get_stock_data; print(get_stock_data('9988.HK','2026-10-20','2026-11-04')[:750])"
```

**港股官方公告与三大报表（按披露时间过滤）**

对 `.HK` 股票，`get_news` 查询 HKEXnews 披露易的**官方公告标题、发布时间和原始 PDF 链接**；三个财报工具和 `get_fundamentals` 查询 AKShare 东方财富的港股长表科目，然后**仅对有对应报告期、并能核验官方披露时间的报表放行**。输入的股票代码会自动映射到港交所内部的 `stockId`，无需手工查表。

为避免同日收盘前后时点混淆，财务报表的披露日期必须**严格早于**指定分析日；对无法确认报告期（包括官方 PDF 前两页无法定位日期）的财务数据返回 `NO_DATA_AVAILABLE`，不会根据 `REPORT_DATE` 猜测公告日。

特别注意：东方财富接口提供的是**当前数据库的数值**，可能发生历史追溯修订；HKEX 原始 PDF 链接提供可核验的首次披露证据，但**并不能保证第三方数值是当时首次披露的原值**。输出保留该警示和计量单位不确定性。HKEXnews 的标题检索是网站使用的非正式接口，真实可用性还需在使用网络中验证。

可不调用 LLM 分别测试：

```powershell
python -c "from tradingagents.agents.tools import get_news; print(get_news.func('9988.HK','2026-08-01','2026-11-04')[:1800])"
python -c "from tradingagents.agents.tools import get_income_statement; print(get_income_statement.func('9988.HK','quarterly','2026-11-04')[:1800])"
```



**3. 启动**

```powershell
trader
```

CLI 会引导选择标的、分析日期、分析师和研究深度。运行过程中会展示 Agent 执行进展；完成后可保存研究报告。

## 使用方式

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

## 项目结构

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
