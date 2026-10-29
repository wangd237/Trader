"""Offline Phase 0 coverage experiment using the real Trader Graph."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel, _graph
from tradingagents.evaluation.phase0_probe import Phase0TraceProbe


@pytest.mark.unit
def test_phase0_observes_graph_llm_tool_and_state_events(tmp_path, monkeypatch, offline):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel())
    probe = Phase0TraceProbe()
    config = graph.propagator.get_graph_args(callbacks=[probe])["config"]

    for namespace, mode, payload in graph.graph.stream(
        graph.create_run_state("NVDA", TRADE_DATE),
        config=config,
        stream_mode=["tasks", "updates"],
        subgraphs=True,
    ):
        probe.record_graph_event(namespace, mode, payload)

    coverage = probe.coverage()
    counts = coverage["counts"]
    for kind in (
        "llm_start", "llm_end", "tool_start", "tool_end",
        "graph_task_start", "graph_task_end", "state_update",
    ):
        assert counts.get(kind, 0) > 0, (kind, coverage)
    assert coverage["llm_unmatched_start"] == 0
    assert coverage["tool_unmatched_start"] == 0
    assert any("Market Analyst" in ns for ns in coverage["subgraph_namespaces"])
    assert any(
        "final_trade_decision" in event["state_fields"]
        for event in probe.events
        if event["kind"] == "state_update"
    )
    assert any(
        event["tool_requests"]
        for event in probe.events
        if event["kind"] == "llm_end"
    )

    trace_path = tmp_path / "phase0_trace.jsonl"
    probe.write_jsonl(trace_path)
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == len(probe.events)
    print("PHASE0_COVERAGE=" + json.dumps(coverage, ensure_ascii=False))


@pytest.mark.unit
def test_phase0_reports_unmatched_callbacks_without_logging_sensitive_values():
    probe = Phase0TraceProbe()
    probe.on_tool_start(
        {"name": "get_stock_data"},
        "secret tool input must not be logged",
        run_id=uuid4(),
        parent_run_id=uuid4(),
    )
    assert probe.coverage()["tool_unmatched_start"] == 1
    assert "secret tool input" not in json.dumps(probe.events)
