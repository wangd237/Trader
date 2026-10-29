"""Offline Phase 0 coverage experiment using the real Trader Graph."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from tests import test_graph_end_to_end as e2e
from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel, _graph
from tradingagents.evaluation.phase0_probe import Phase0TraceProbe

# Reuse the existing offline provider fixture; imported fixtures need a module-level name.
offline = e2e.offline


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
    # Print BEFORE assertions, so failed CI preserves diagnostic evidence.
    print("PHASE0_COVERAGE=" + json.dumps(coverage, ensure_ascii=False))
    counts = coverage["counts"]
    for kind in (
        "llm_start", "llm_end", "tool_start", "tool_end",
        "graph_task_start", "graph_task_end", "state_update",
    ):
        assert counts.get(kind, 0) > 0, (kind, coverage)
    expected_nodes = {
        "Market Analyst", "Sentiment Analyst", "News Analyst",
        "Fundamentals Analyst", "Memory Log",
        "Bull Researcher", "Bear Researcher", "Research Manager", "Trader",
        "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst",
        "Portfolio Manager",
    }
    assert expected_nodes <= set(coverage["top_level_completed_nodes"]), coverage

    # Full two-way pairing: not just starts without ends, but also orphan
    # ends, duplicated spans and missing run/task IDs.
    for kind in ("graph_pairing", "llm_pairing", "tool_pairing"):
        assert all(n == 0 for n in coverage[kind].values()), (kind, coverage[kind])

    # Verify real model-request -> ToolNode result evidence, scoped by analyst
    # subgraph (tool-call IDs can repeat in independent analyst subgraphs).
    linkage = coverage["tool_call_linkage"]
    assert linkage["requested"] > 0
    assert linkage["matched"] == linkage["requested"] == linkage["result_messages"]
    assert linkage["unmatched_requests"] == 0
    assert linkage["unmatched_results"] == 0
    assert linkage["unscoped_requests"] == 0
    assert linkage["unscoped_results"] == 0
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


@pytest.mark.unit
def test_phase0_reports_unmatched_callbacks_without_logging_sensitive_values():
    probe = Phase0TraceProbe()
    probe.on_tool_start(
        {"name": "get_stock_data"},
        "secret tool input must not be logged",
        run_id=uuid4(),
        parent_run_id=uuid4(),
    )
    assert probe.coverage()["tool_pairing"]["missing_end"] == 1
    assert "secret tool input" not in json.dumps(probe.events)


@pytest.mark.unit
def test_phase0_pairing_reports_orphan_ends_and_duplicates():
    probe = Phase0TraceProbe()
    run_id = uuid4()
    probe.on_tool_end("data", run_id=run_id)
    assert probe.coverage()["tool_pairing"]["missing_start"] == 1

    probe.on_tool_start({"name": "get_stock_data"}, "secret", run_id=run_id)
    probe.on_tool_start({"name": "get_stock_data"}, "secret", run_id=run_id)
    paired = probe.coverage()["tool_pairing"]
    assert paired["missing_start"] == 0
    assert paired["missing_end"] == 1
    assert paired["duplicate_start"] == 1
