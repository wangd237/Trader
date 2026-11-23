# Trader

**Trader** 是基于 LangGraph 构建的多智能体金融市场研究框架。多个 Agent 协作完成市场分析、多空辩论、交易方案制定和风险评估，最终生成投资研究结论与结构化报告。

项目面向**研究分析与辅助决策**，不执行真实账户交易。

## 核心功能

- **多 Agent 协作**：市场、基本面、新闻和情绪分析师并行研究，结合多空辩论与风险讨论形成最终决策。
- **LangGraph 工作流**：支持条件路由、独立工具调用子图、状态传递和可选的检查点恢复。
- **多市场数据接入**：支持 A 股、港股及美股，通过统一路由自动选择行情、财报和新闻数据源。
- **研究记忆与评测**：记录历史决策与反思，支持执行轨迹采集、规则评测和历史回测。
- **多种使用方式**：提供交互式 CLI、非交互命令及 Python API，输出 Markdown / HTML 研究报告。

## 工作流程

```text
Ticker / 分析日期 / 可选持仓
             │
             ▼
市场 / 基本面 / 新闻 / 情绪分析师（并行）
             │
             ▼
       多空研究员辩论
             │
             ▼
        Research Manager
             │
             ▼
            Trader
             │
             ▼
           风险讨论
             │
             ▼
        Portfolio Manager
             │
             ▼
       最终决策与研究报告
```

各 Agent 由 LangGraph 编排，底层通过统一数据路由调用不同市场的数据提供商。

## 快速开始

**环境要求**：Python 3.11+、Git、可用的 LLM API。

### 安装

```powershell
git clone https://github.com/wangd237/Trader.git
cd Trader
conda create -n trader python=3.11 -y
conda activate trader
python -m pip install -e ".[ashare]"
Copy-Item .env.example .env
```

### 配置

编辑根目录的 `.env`：

```dotenv
LLM_API_KEY=your_api_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
TRADINGAGENTS_OUTPUT_LANGUAGE=Chinese
TRADINGAGENTS_RESULTS_DIR=./results
```

### 运行

交互式启动：

```powershell
trader
```

**示例**（指定标的、日期及分析师）：

```powershell
trader --ticker 600519.SH --date 2026-09-30 --analysts market,fundamentals,news,social --save --no-show
```

运行完成后，可在配置的结果目录中查看研究报告与执行记录。更多命令参数可通过 `trader --help` 查看。

## 项目结构

```text
Trader/
├── cli/                  # CLI 入口与交互界面
├── tradingagents/
│   ├── agents/           # 分析师、研究员、交易员和风控 Agent
│   ├── graph/            # LangGraph 工作流与状态管理
│   ├── dataflows/        # 数据源路由与市场适配
│   ├── llm_clients/      # LLM 客户端
│   ├── memory/           # 历史决策与经验记忆
│   └── backtest.py       # 历史回测
├── tests/                # 测试
├── scripts/              # 辅助脚本
└── pyproject.toml        # 项目依赖与配置
```
