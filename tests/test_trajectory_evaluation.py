"""Offline tests for Phase 1: real Trader graph, simulated LLM and vendors."""

from __future__ import annotations

import json

import pytest

from tests import test_graph_end_to_end as e2e
from tests.test_graph_end_to_end import TRADE_DATE, LoopingModel, ScriptedModel, _graph
from tradingagents.evaluation.collector import (
    TrajectoryCollector,
    _date_errors,
    collect_trajectory,
    load_trace,
)
from tradingagents.evaluation.rules import evaluate_events

offline = e2e.offline


@pytest.mark.unit
def test_collector_offline_full_graph_produces_reproducible_evaluation(
    tmp_path, monkeypatch, offline
):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel())
    report = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="offline_full_graph", attempt_id="1", output_dir=tmp_path / "eval",
    )

    print("PHASE1_SUMMARY=" + json.dumps({
        "status": report["status"],
        "rules": {key: value["status"] for key, value in report["rules"].items()},
        "metrics": report["metrics"],
        "coverage": report["coverage"]["tool_call_linkage"],
    }, ensure_ascii=False))
    assert report["status"] == "pass", report
    assert all(rule["status"] != "fail" for rule in report["rules"].values())
    assert report["rules"]["task_completion"]["status"] == "pass"
    assert report["rules"]["graph_route"]["status"] == "pass"
    assert report["metrics"]["tool_calls"] == 12
    assert report["metrics"]["candidate_duplicate_tool_calls"] == 0

    run_dir = tmp_path / "eval" / report["run_id"]
    saved = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert saved == report
    events = load_trace(run_dir / "trace.jsonl")
    assert evaluate_events(events) == report
    assert all(e.get("event_id") for e in events)
    assert len({e["event_id"] for e in events}) == len(events)
    disk_rows = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["run_id"] == report["run_id"] for row in disk_rows)
    assert not any("raw_args" in row or "messages" in row for row in disk_rows)
    assert graph.memory_log.load_entries() == [], "diagnostic runner must not record decisions"


@pytest.mark.unit
def test_collector_captures_failure_and_writes_trace(tmp_path, monkeypatch, offline):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(fail_at=12))
    result = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="interrupted", attempt_id="1", output_dir=tmp_path / "eval",
    )
    assert result["status"] == "fail"
    assert result["rules"]["task_completion"]["status"] == "fail"
    assert (tmp_path / "eval" / result["run_id"] / "trace.jsonl").exists()


@pytest.mark.unit
def test_budget_rounds_are_not_conflated_with_tool_calls(tmp_path, monkeypatch, offline):
    graph = _graph(
        tmp_path, monkeypatch, LoopingModel(structured=True), max_tool_rounds=2
    )
    report = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="wrapup", attempt_id="1", output_dir=tmp_path / "eval",
    )
    assert report["rules"]["tool_round_budget"]["status"] == "pass", report
    assert report["metrics"]["tool_calls"] >= 6


@pytest.mark.unit
def test_requested_dates_and_reverse_window_are_flagged():
    assert _date_errors(
        {"start_date": "2026-01-02", "end_date": "2026-01-10"}, TRADE_DATE
    ) == ["end_date:after_trade_date"]
    assert _date_errors(
        {"start_date": "2026-01-08", "end_date": "2026-01-02"}, TRADE_DATE
    ) == ["window:reversed"]
    assert _date_errors({"curr_date": "bad"}, TRADE_DATE) == ["curr_date:invalid_date"]


@pytest.mark.unit
def test_invalid_tool_intent_and_duplicates_are_reported_not_autoscored():
    collector = TrajectoryCollector(
        case_id="synthetic", attempt_id="1", trade_date=TRADE_DATE,
        selected_analysts=("market",), max_tool_rounds=2,
    )
    for _ in range(2):
        collector._emit(
            "tool_intent",
            scope="Market Analyst:synthetic",
            agent="Market Analyst",
            tool_name="get_stock_data",
            arg_fingerprint="same_run_only",
            authorized=True,
            schema_valid=False,
            date_errors=["end_date:after_trade_date"],
            malformed=False,
        )
    collector.finish({}, RuntimeError("injected failure"))
    report = evaluate_events(collector.events)
    assert report["rules"]["model_tool_intent"]["status"] == "fail"
    assert report["metrics"]["candidate_duplicate_tool_calls"] == 1
    assert report["candidate_duplicates"][0]["status"] == "candidate_requires_semantic_review"


@pytest.mark.unit
def test_diagnostic_collector_rejects_checkpoint_mode(tmp_path, monkeypatch):
    graph = _graph(
        tmp_path, monkeypatch, ScriptedModel(), checkpoint_enabled=True
    )
    with pytest.raises(ValueError, match="checkpoint_enabled"):
        collect_trajectory(
            graph, "NVDA", TRADE_DATE, case_id="bad_config",
            attempt_id="1", output_dir=tmp_path,
        )
