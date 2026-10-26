from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# DEFAULT_CONFIG 已应用 TRADINGAGENTS_* 环境变量覆盖。
config = DEFAULT_CONFIG.copy()

# 通过自定义配置初始化。
ta = TradingAgentsGraph(debug=True, config=config)

# 向前执行 Graph 的传播流程。
_, decision = ta.propagate("NVDA", "2026-09-01")
print(decision)
