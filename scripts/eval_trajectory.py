"""Re-evaluate one or more saved Phase 1 JSONL traces without an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingagents.evaluation.collector import load_trace
from tradingagents.evaluation.rules import evaluate_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved Trader trajectory JSONL files")
    parser.add_argument("paths", nargs="+", type=Path, help="trace.jsonl files or result directories")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if a rule fails")
    args = parser.parse_args()
    traces = []
    for path in args.paths:
        if path.is_file():
            traces.append(path)
        elif path.is_dir():
            traces.extend(sorted(path.rglob("trace.jsonl")))
        else:
            parser.error(f"not found: {path}")
    if not traces:
        parser.error("no trace.jsonl files found")
    failed = 0
    for path in traces:
        result = evaluate_events(load_trace(path))
        destination = path.parent / "evaluation.json"
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        failed += result["status"] == "fail"
        print(
            json.dumps({
                "run_id": result["run_id"],
                "case_id": result["case_id"],
                "status": result["status"],
                "llm_calls": result["metrics"]["llm_calls"],
                "tool_calls": result["metrics"]["tool_calls"],
                "candidate_duplicates": result["metrics"]["candidate_duplicate_tool_calls"],
                "report": str(destination),
            }, ensure_ascii=False)
        )
    print(f"evaluated={len(traces)} failed={failed}")
    return int(args.strict and failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
