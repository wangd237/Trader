"""Negative evidence and dynamic routing regressions for Phase 1."""

from __future__ import annotations

import json

import pytest

from tests import test_graph_end_to_end as e2e
from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel, _graph
from tradingagents.evaluation.collector import collect_trajectory, load_trace
from tradingagents.evaluation.rules import evaluate_events

offline = e2e.offline


@pytest.mark.unit
def test_missing_tool_intents_cannot_score_pass(tmp_path, monkeypatch, offline):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel())
    original = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="evidence_integrity", attempt_id="1", output_dir=tmp_path / "eval",
    )
    assert original["status"] == "pass"
    path = tmp_path / "eval" / original["run_id"] / "trace.jsonl"
    events = load_trace(path)
    without_intents = [e for e in events if e["kind"] != "tool_intent"]
    invalid = evaluate_events(without_intents)
    assert invalid["status"] == "fail"
    assert invalid["rules"]["evidence_completeness"]["status"] == "fail"
    assert "unobserved_model_tool_intents:12" in invalid["rules"]["evidence_completeness"]["details"]

    without_llm = [
        e for e in events if e["kind"] not in {"llm_start", "llm_end"}
    ]
    invalid = evaluate_events(without_llm)
    assert invalid["status"] == "fail"
    assert "missing_evidence:llm_start" in invalid["rules"]["evidence_completeness"]["details"]

    without_graph = [
        e for e in events if not e["kind"].startswith("graph_task_")
    ]
    invalid = evaluate_events(without_graph)
    assert invalid["status"] == "fail"
    assert "missing_evidence:graph_task_end" in invalid["rules"]["evidence_completeness"]["details"]
    print("PHASE1_NEGATIVE=" + json.dumps({
        "missing_intents": "fail",
        "missing_llm_callbacks": "fail",
        "missing_graph_events": "fail",
    }))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("debate_rounds", "risk_rounds"),
    [(0, 0), (2, 2)],
    ids=["zero-round-shortcuts", "multiple-debate-and-risk-rounds"],
)
def test_dynamic_graph_route_matches_configuration(
    tmp_path, monkeypatch, offline, debate_rounds, risk_rounds
):
    graph = _graph(
        tmp_path, monkeypatch, ScriptedModel(),
        max_debate_rounds=debate_rounds,
        max_risk_discuss_rounds=risk_rounds,
    )
    report = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="dynamic_route", attempt_id="1", output_dir=tmp_path / "eval",
    )
    assert report["status"] == "pass", report
    assert report["rules"]["graph_route"]["status"] == "pass", report


@pytest.mark.unit
def test_missing_route_node_fails_without_requiring_unconfigured_roles(
    tmp_path, monkeypatch, offline
):
    graph = _graph(
        tmp_path, monkeypatch, ScriptedModel(),
        max_debate_rounds=0, max_risk_discuss_rounds=0,
    )
    result = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="route_missing", attempt_id="1", output_dir=tmp_path / "eval",
    )
    events = load_trace(
        tmp_path / "eval" / result["run_id"] / "trace.jsonl"
    )
    assert evaluate_events(events)["rules"]["graph_route"]["status"] == "pass"
    truncated = [
        e for e in events
        if not (e["kind"] == "graph_task_end" and e.get("node") == "Trader")
    ]
    invalid = evaluate_events(truncated)
    assert invalid["rules"]["graph_route"]["status"] == "fail"


@pytest.mark.unit
def test_initialization_failure_persists_a_failed_trace(tmp_path, monkeypatch):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel())

    def fail_initialization(*args, **kwargs):
        raise RuntimeError("simulated initialization error")

    monkeypatch.setattr(graph, "create_run_state", fail_initialization)
    report = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="init_failure", attempt_id="1", output_dir=tmp_path / "eval",
    )
    assert report["status"] == "fail"
    assert report["rules"]["task_completion"]["status"] == "fail"
    assert report["rules"]["evidence_completeness"]["status"] == "not_applicable"
    path = tmp_path / "eval" / report["run_id"] / "trace.jsonl"
    events = load_trace(path)
    final = next(e for e in events if e["kind"] == "evaluation_end")
    assert final["error_type"] == "RuntimeError"
    assert final["error_stage"] == "initialization"
    assert "simulated initialization error" not in path.read_text(encoding="utf-8")
