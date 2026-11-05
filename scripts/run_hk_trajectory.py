"""Live A/HK Agent trajectory diagnostic using the existing Phase-1 evaluator.

Example (makes paid LLM calls and contacts live market vendors):
python scripts/run_hk_trajectory.py --ticker 9988.HK --date 2026-11-04 --confirm-live
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

# Load API credentials before importing the default LLM configuration.
load_dotenv()

from tradingagents.dataflows.symbols import safe_ticker_component  # noqa: E402
from tradingagents.dataflows.vendors.hk import is_hk  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.evaluation.collector import collect_trajectory  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real HK analysts + Trader graph and save safe trajectory metrics",
    )
    parser.add_argument("--ticker", default="9988.HK", help="HKEX ticker, e.g. 9988.HK")
    parser.add_argument("--date", default="2026-11-04", help="Analysis date YYYY-MM-DD")
    parser.add_argument("--analysts", default="market,news,fundamentals")
    parser.add_argument("--attempt", default="01")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="Acknowledge real network calls and potentially billable LLM usage",
    )
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("pass --confirm-live to acknowledge network requests and LLM cost")
    if not is_hk(args.ticker):
        parser.error("--ticker must be a Hong Kong stock such as 9988.HK")
    from datetime import date

    try:
        analysis_date = date.fromisoformat(args.date)
    except ValueError:
        parser.error("--date must be YYYY-MM-DD")
    if analysis_date > date.today():
        parser.error("--date must not be in the future")
    if not args.attempt.isalnum() or args.max_tool_rounds < 1:
        parser.error("--attempt must be alphanumeric and --max-tool-rounds must be >= 1")

    analysts = tuple(s.strip() for s in args.analysts.split(",") if s.strip())
    supported = {"market", "news", "fundamentals", "social"}
    if not analysts or len(analysts) != len(set(analysts)) or set(analysts) - supported:
        parser.error("--analysts must be distinct names from market,news,fundamentals,social")

    ticker = safe_ticker_component(args.ticker.upper())
    case_id = f"{ticker}_{analysis_date}_hk_live"
    root = Path("results") / "trajectory_hk_live" / case_id / args.attempt
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
    })
    # No benchmark fixture: this runner actually tests the configured HK data
    # vendors, unlike the existing frozen-price baseline.
    config.pop("benchmark_ohlcv_csv", None)
    manifest = {
        "case_id": case_id,
        "attempt_id": args.attempt,
        "ticker": ticker,
        "analysis_date": str(analysis_date),
        "analysts": list(analysts),
        "llm_provider": config["llm_provider"],
        "quick_think_llm": config["quick_think_llm"],
        "deep_think_llm": config["deep_think_llm"],
        "max_tool_rounds": args.max_tool_rounds,
        "data_vendors": dict(config["data_vendors"]),
        "data_mode": "live_hk",
        "caution": (
            "Trace evaluation checks structure/tool-call integrity, not whether "
            "LLM conclusions correctly interpret financial numbers."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    graph = TradingAgentsGraph(selected_analysts=analysts, config=config)
    report = collect_trajectory(
        graph, ticker, str(analysis_date),
        case_id=case_id, attempt_id=args.attempt, output_dir=root / "traces",
    )
    print(json.dumps({
        "status": report["status"],
        "run_id": report["run_id"],
        "metrics": report["metrics"],
        "rules": {
            key: {"status": val["status"], "details": val["details"]}
            for key, val in report["rules"].items()
        },
        "trace_dir": str(root / "traces" / report["run_id"]),
        "manual_review": (
            "Check the agent's actual financial claims, currency, report dates "
            "and HKEX citations separately. This JSONL omits tool output contents."
        ),
    }, ensure_ascii=False, indent=2))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
