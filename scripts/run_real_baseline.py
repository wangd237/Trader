"""Real-LLM single Market Analyst smoke test using pinned historical OHLCV.

Example:
python scripts/run_real_baseline.py --csv E:/market_data/NVDA.csv --source "historical CSV export"
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from tradingagents.dataflows.config import run_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evaluation.collector import collect_trajectory
from tradingagents.evaluation.frozen_market import fixture_sha256, load_frozen_ohlcv
from tradingagents.graph.trading_graph import TradingAgentsGraph


def main() -> int:
    parser = argparse.ArgumentParser(description="Real LLM + offline frozen OHLCV baseline")
    parser.add_argument("--csv", required=True, type=Path, help="Local historical CSV with Date/Open/High/Low/Close/Volume")
    parser.add_argument("--source", required=True, help="Source/provenance of the CSV (recorded, not independently verified)")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--company", default="NVIDIA Corporation")
    parser.add_argument("--trade-date", default="2026-09-30")
    parser.add_argument("--attempt", default="01")
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    args = parser.parse_args()

    if args.max_tool_rounds < 1 or not args.attempt.isalnum():
        parser.error("--max-tool-rounds must be >=1 and --attempt must be alphanumeric")
    source = args.csv.expanduser().resolve()
    if not source.is_file():
        parser.error(f"CSV file not found: {source}")
    digest = fixture_sha256(source)
    case_id = f"{args.ticker.upper()}_{args.trade_date}_market"
    root = Path("results/trajectory_baseline") / case_id / args.attempt
    root.mkdir(parents=True, exist_ok=True)

    config = deepcopy(DEFAULT_CONFIG)
    config.update({
        "checkpoint_enabled": False,
        "data_cache_dir": str(root / "cache"),
        "memory_log_path": str(root / "memory.md"),
        "results_dir": str(root / "reports"),
        "max_tool_rounds": args.max_tool_rounds,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "benchmark_ohlcv_csv": str(source),
        "benchmark_ohlcv_sha256": digest,
        "benchmark_symbol": args.ticker.upper(),
        "benchmark_company_name": args.company,
    })

    # Check data before the first paid LLM call. Never silently fetch Yahoo.
    with run_config(config):
        data = load_frozen_ohlcv(args.ticker, args.trade_date)
    if len(data) < 200:
        parser.error(
            f"CSV has only {len(data)} OHLCV rows before {args.trade_date}; "
            "provide at least 200 trading rows for the 200-day SMA"
        )

    manifest = {
        "case_id": case_id,
        "attempt_id": args.attempt,
        "ticker": args.ticker.upper(),
        "company_name": args.company,
        "trade_date": args.trade_date,
        "csv_path": str(source),
        "csv_sha256": digest,
        "csv_source_user_supplied": args.source,
        "available_rows": len(data),
        "first_row": str(data["Date"].min().date()),
        "last_row": str(data["Date"].max().date()),
        "llm_provider": config.get("llm_provider"),
        "model_quick": config.get("quick_think_llm"),
        "model_deep": config.get("deep_think_llm"),
        "max_tool_rounds": args.max_tool_rounds,
    }
    (root / "fixture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Frozen fixture:", source, "SHA256:", digest)
    print("OHLCV rows:", len(data), "as of", args.trade_date)

    graph = TradingAgentsGraph(selected_analysts=("market",), config=config)
    report = collect_trajectory(
        graph, args.ticker, args.trade_date,
        case_id=case_id, attempt_id=args.attempt, output_dir=root / "traces",
    )

    print("\n=== Real LLM Frozen Baseline ===")
    print("run_id:", report["run_id"])
    print("status:", report["status"])
    print("metrics:", report["metrics"])
    for name, result in report["rules"].items():
        print(name, result["status"], result["details"])
    print("output:", root / "traces" / report["run_id"])
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
