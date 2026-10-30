
from copy import deepcopy
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.evaluation.collector import collect_trajectory


# 1. 固定任务
ticker = "NVDA"
trade_date = "2026-09-30"
case_id = "nvda_market_20260930"
attempt_id = "01"

# 2. 隔离每次运行的目录
root = Path("results/trajectory_baseline") / case_id / attempt_id
root.mkdir(parents=True, exist_ok=True)

# 3. 固定 Harness 配置
config = deepcopy(DEFAULT_CONFIG)
config.update({
    "checkpoint_enabled": False,
    "data_cache_dir": str(root / "cache"),
    "memory_log_path": str(root / "memory.md"),
    "results_dir": str(root / "reports"),
    "max_tool_rounds": 3,
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
})

# 4. 初始化真实 LLM Graph
graph = TradingAgentsGraph(
    selected_analysts=("market",),
    config=config,
)

# 5. 真实执行并采集轨迹
report = collect_trajectory(
    graph,
    ticker,
    trade_date,
    case_id=case_id,
    attempt_id=attempt_id,
    output_dir=root / "traces",
)

# 6. 输出规则评分
print("\n=== Real LLM Baseline ===")
print("Run ID:", report["run_id"])
print("Status:", report["status"])
print("Metrics:", report["metrics"])

for name, result in report["rules"].items():
    print(name, result["status"], result["details"])

print("Output:", root / "traces" / report["run_id"])
