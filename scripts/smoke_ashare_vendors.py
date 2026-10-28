"""Real A-share provider smoke test (does not invoke any LLM or Agent)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tradingagents.dataflows.router import route_to_vendor

SAMPLES = ("600519.SH", "000001.SZ", "300750.SZ")


def run_case(method: str, *args):
    try:
        result = route_to_vendor(method, *args)
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, str) or not result.strip():
        return {"status": "ERROR", "error": "empty response"}
    if result.startswith(("DATA_UNAVAILABLE", "NO_DATA_AVAILABLE", "NO_NEWS_AVAILABLE")):
        return {"status": "UNAVAILABLE", "preview": result[:300]}
    return {"status": "OK", "characters": len(result), "preview": result[:300]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ashare_stage_b_smoke.json")
    args = parser.parse_args()
    cases = []
    for ticker in SAMPLES:
        cases.extend([
            (ticker, "get_stock_data", (ticker, "2026-09-01", "2026-09-30")),
            (ticker, "get_income_statement", (ticker, "quarterly", "2026-09-30")),
            (ticker, "get_balance_sheet", (ticker, "quarterly", "2026-09-30")),
            (ticker, "get_cashflow", (ticker, "quarterly", "2026-09-30")),
            (ticker, "get_fundamentals", (ticker, "2026-09-30")),
            (ticker, "get_indicators", (ticker, "rsi", "2026-09-30", 14)),
        ])
    cases.extend([
        ("600519.SH", "get_news", ("600519.SH", "2026-08-14", "2026-08-16")),
        ("000001.SZ", "get_news", ("000001.SZ", "2026-09-01", "2026-09-30")),
        ("300750.SZ", "get_news", ("300750.SZ", "2026-09-01", "2026-09-30")),
    ])
    results = []
    for ticker, method, params in cases:
        value = run_case(method, *params)
        results.append({"ticker": ticker, "method": method, **value})
        print(ticker, method, value["status"], value.get("error", ""))
    payload = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "llm_used": False,
        "results": results,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
