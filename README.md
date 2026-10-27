# Trader

**Trader** 是一个多智能体金融市场研究框架。它将行情、基本面、新闻和市场情绪交给不同的分析 Agent，再经过多空辩论、交易计划和风险评议，形成最终研究结论。

> **声明**：模型可能产生错误分析；输出不构成投资建议，也不应直接用于真实资金的自动交易。

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

运行层使用 LangGraph 编排与工具接口，包含多模型提供商接入、数据源路由、记忆/检查点、报告和历史回测。

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

打开 .env，只需填写三个 **LLM** 字段：

~~~env
LLM_API_KEY=你的密钥
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
~~~
并配置报告保存地址和输出语言：
~~~env
TRADINGAGENTS_OUTPUT_LANGUAGE=Chinese
TRADINGAGENTS_RESULTS_DIR=自定义报告保存地址
~~~



## 使用

~~~powershell
# 交互式引导选择标的、分析日期和分析师
trader

# 查看可用参数
trader --help

# 配置上述三个 LLM 字段后，可使用无交互选项
trader --ticker NVDA --date 2026-10-20 --analysts market,news --save --no-show

# 可选：回测 (会消耗模型调用额度)
trader backtest NVDA --start 2026-09-01 --end 2026-09-15 --every 7
~~~



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

