"""Offline regression tests for the conservative runtime tool-loop guard."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests import test_graph_end_to_end as e2e
from tradingagents.agents.analysts.turn import TOOL_LOOP_WARNING
from tradingagents.graph.tool_loop_guard import (
    cached_historical_tool_messages,
    consecutive_identical_tool_rounds,
    repeats_warned_tool_request,
)

offline = e2e.offline


def _round(name="get_stock_data", args=None, result="same", call_id="id-1"):
    call = AIMessage(content="", tool_calls=[{
        "name": name, "args": args if args is not None else {"ticker": "NVDA"},
        "id": call_id,
    }])
    reply = ToolMessage(content=result, tool_call_id=call_id)
    return [call, reply]


@pytest.mark.unit
def test_identical_tool_rounds_are_detected_ignoring_call_ids():
    messages = [
        HumanMessage(content="research"),
        *_round(call_id="a"),
        *_round(call_id="b"),
        *_round(call_id="c"),
    ]
    assert consecutive_identical_tool_rounds(messages) == 3


@pytest.mark.unit
def test_new_arguments_or_new_data_resets_streak():
    messages = [
        *_round(call_id="a"),
        *_round(args={"ticker": "MSFT"}, call_id="b"),
        *_round(args={"ticker": "MSFT"}, result="new result", call_id="c"),
    ]
    assert consecutive_identical_tool_rounds(messages) == 1


@pytest.mark.unit
def test_incomplete_or_malformed_round_is_not_treated_as_stagnation():
    messages = [*_round(call_id="a"), AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "b"}
    ])]
    assert consecutive_identical_tool_rounds(messages) == 0


@pytest.mark.unit
def test_unmatched_tool_result_breaks_streak():
    messages = [*_round(call_id="a"), AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "b"}
    ]), ToolMessage(content="same", tool_call_id="wrong")]
    assert consecutive_identical_tool_rounds(messages) == 0


@pytest.mark.unit
def test_multi_tool_round_keeps_order_and_compares_every_result():
    def round_pair(prefix, result_2="B"):
        return [
            AIMessage(content="", tool_calls=[
                {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": prefix + "1"},
                {"name": "get_indicators", "args": {"ticker": "NVDA"}, "id": prefix + "2"},
            ]),
            ToolMessage(content=result_2, tool_call_id=prefix + "2"),
            ToolMessage(content="A", tool_call_id=prefix + "1"),
        ]

    assert consecutive_identical_tool_rounds([
        *round_pair("a"), *round_pair("b"),
    ]) == 2
    assert consecutive_identical_tool_rounds([
        *round_pair("a"), *round_pair("b", "changed"),
    ]) == 1


@pytest.mark.unit
def test_runtime_stops_before_maximum_when_repeated_rounds_stagnate(tmp_path, monkeypatch, offline):
    from tests.test_graph_end_to_end import LoopingModel, _graph

    model = LoopingModel(structured=True)
    graph = _graph(
        tmp_path, monkeypatch, model,
        max_tool_rounds=8, max_stagnant_tool_rounds=3,
    )
    state, rating = graph.propagate("NVDA", "2026-01-09")
    # Each analyst gets 3 identical results, one warning, then a blocked 4th request.
    assert len(model.tool_turns) == 12
    assert rating == "Overweight"
    assert state["market_report"]


@pytest.mark.unit
def test_runtime_can_disable_early_stop(tmp_path, monkeypatch, offline):
    from tests.test_graph_end_to_end import LoopingModel, _graph

    model = LoopingModel(structured=True)
    graph = _graph(
        tmp_path, monkeypatch, model,
        max_tool_rounds=4, max_stagnant_tool_rounds=0,
    )
    graph.propagate("NVDA", "2026-01-09")
    assert len(model.tool_turns) == 12



@pytest.mark.unit
def test_cache_reuses_identical_historical_request_with_new_tool_call_id():
    history = [*_round(call_id="a"), AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "b"},
    ])]
    replay = cached_historical_tool_messages(history, historical_run=True)
    assert replay is not None and len(replay) == 1
    assert replay[0].tool_call_id == "b"
    assert replay[0].content == "same"


@pytest.mark.unit
def test_cache_respects_history_scope_and_dynamic_tools():
    history = [*_round(call_id="a"), AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "b"},
    ])]
    assert cached_historical_tool_messages(history, historical_run=False) is None
    news = [*_round(name="get_news", call_id="a"), AIMessage(content="", tool_calls=[
        {"name": "get_news", "args": {"ticker": "NVDA"}, "id": "b"},
    ])]
    assert cached_historical_tool_messages(news, historical_run=True) is None


@pytest.mark.unit
def test_cache_never_replays_failed_or_unavailable_tool_response():
    old = _round(call_id="a")
    old[1] = ToolMessage(content="DATA_UNAVAILABLE: vendor down", tool_call_id="a")
    pending = AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "b"},
    ])
    assert cached_historical_tool_messages([*old, pending], historical_run=True) is None
    old[1] = ToolMessage(content="same", tool_call_id="a", status="error")
    assert cached_historical_tool_messages([*old, pending], historical_run=True) is None


@pytest.mark.unit
def test_cache_requires_same_arguments():
    history = [*_round(call_id="a"), AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "MSFT"}, "id": "b"},
    ])]
    assert cached_historical_tool_messages(history, historical_run=True) is None


@pytest.mark.unit
def test_graph_reuses_historical_stock_data_without_reexecuting(tmp_path, monkeypatch, offline):
    from tests.test_graph_end_to_end import LoopingModel, _graph
    from tradingagents.dataflows import router

    executions = []
    for vendor in router.VENDOR_METHODS["get_stock_data"]:
        monkeypatch.setitem(
            router.VENDOR_METHODS["get_stock_data"], vendor,
            lambda *args, **kwargs: executions.append(args) or "historical prices",
        )
    model = LoopingModel(structured=True)
    graph = _graph(
        tmp_path, monkeypatch, model,
        max_tool_rounds=8, max_stagnant_tool_rounds=3,
        reuse_historical_tool_results=True,
    )
    state, rating = graph.propagate("NVDA", "2026-01-09")
    assert len(executions) == 1  # same market query was requested three times
    assert len(model.tool_turns) == 12
    assert state["market_report"]
    assert rating == "Overweight"


@pytest.mark.unit
def test_graph_can_disable_pre_execution_cache(tmp_path, monkeypatch, offline):
    from tests.test_graph_end_to_end import LoopingModel, _graph
    from tradingagents.dataflows import router

    executions = []
    for vendor in router.VENDOR_METHODS["get_stock_data"]:
        monkeypatch.setitem(
            router.VENDOR_METHODS["get_stock_data"], vendor,
            lambda *args, **kwargs: executions.append(args) or "historical prices",
        )
    graph = _graph(
        tmp_path, monkeypatch, LoopingModel(structured=True),
        max_tool_rounds=4, max_stagnant_tool_rounds=3,
        reuse_historical_tool_results=False,
    )
    graph.propagate("NVDA", "2026-01-09")
    assert len(executions) == 3



@pytest.mark.unit
def test_latest_failed_request_blocks_older_successful_cache():
    messages = [
        *_round(call_id="a"),
        AIMessage(content="", tool_calls=[
            {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "b"},
        ]),
        ToolMessage(content="DATA_UNAVAILABLE: vendor down", tool_call_id="b"),
        AIMessage(content="", tool_calls=[
            {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "c"},
        ]),
    ]
    assert cached_historical_tool_messages(messages, historical_run=True) is None



@pytest.mark.unit
def test_repeated_wrapup_prompt_names_tool_and_removes_tools(tmp_path, monkeypatch, offline):
    from tests.test_graph_end_to_end import LoopingModel, _graph

    model = LoopingModel(structured=True)
    graph = _graph(
        tmp_path, monkeypatch, model,
        max_tool_rounds=8, max_stagnant_tool_rounds=3,
    )
    graph.propagate("NVDA", "2026-01-09")
    # The shared scripted model also records non-tool researcher/manager calls.
    wrapups = [
        history for history in model.last_turns
        if isinstance(history[-1], HumanMessage)
        and str(history[-1].content).startswith("Tool access is now disabled")
    ]
    assert len(wrapups) == 3
    assert all(
        sum(
            isinstance(m, HumanMessage)
            and str(m.content).startswith(TOOL_LOOP_WARNING)
            for m in history
        ) == 1
        for history in wrapups
    )
    assert any("Repeated tool(s): get_stock_data" in str(h[-1].content) for h in wrapups)
    assert all(
        "repeated requests returned no new data" in str(h[0].content)
        for h in wrapups
    )



@pytest.mark.unit
def test_warning_matches_only_the_immediately_repeated_request():
    warned = [
        *_round(call_id="a"),
        *_round(call_id="b"),
        *_round(call_id="c"),
        HumanMessage(content=TOOL_LOOP_WARNING + " Use another tool."),
    ]
    same = AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "NVDA"}, "id": "d"}
    ])
    different = AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"ticker": "MSFT"}, "id": "d"}
    ])
    assert repeats_warned_tool_request([*warned, same], TOOL_LOOP_WARNING)
    assert not repeats_warned_tool_request([*warned, different], TOOL_LOOP_WARNING)
    assert not repeats_warned_tool_request(
        [*warned, *_round(call_id="x"), same], TOOL_LOOP_WARNING
    )


@pytest.mark.unit
def test_corrected_model_can_write_report_without_forced_wrapup(tmp_path, monkeypatch, offline):
    class ResponsiveModel(e2e.LoopingModel):
        warnings: list = e2e.Field(default_factory=list)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.tools and any(
                isinstance(m, HumanMessage)
                and str(m.content).startswith(TOOL_LOOP_WARNING) for m in messages
            ):
                self.warnings.append(messages)
                return e2e.ChatResult(
                    generations=[e2e.ChatGeneration(message=AIMessage(content=e2e.TEXT))]
                )
            return super()._generate(messages, stop, run_manager, **kwargs)

    model = ResponsiveModel(structured=True)
    graph = e2e._graph(
        tmp_path, monkeypatch, model,
        max_tool_rounds=8, max_stagnant_tool_rounds=3,
    )
    state, rating = graph.propagate("NVDA", "2026-01-09")
    assert len(model.warnings) == 3
    assert len(model.tool_turns) == 9
    assert not any(
        str(h[-1].content).startswith("Tool access is now disabled")
        for h in model.last_turns
    )
    assert state["market_report"] == e2e.TEXT
    assert rating == "Overweight"


@pytest.mark.unit
def test_corrected_model_can_change_tools_and_continue(tmp_path, monkeypatch, offline):
    class SwitchToolModel(e2e.LoopingModel):
        changes: list = e2e.Field(default_factory=list)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            was_warned = any(
                isinstance(m, HumanMessage)
                and str(m.content).startswith(TOOL_LOOP_WARNING) for m in messages
            )
            if self.tools and was_warned:
                if isinstance(messages[-1], ToolMessage):
                    return e2e.ChatResult(
                        generations=[e2e.ChatGeneration(message=AIMessage(content=e2e.TEXT))]
                    )
                other = self.tools[1]
                self.changes.append(other.name)
                call = {
                    "name": other.name, "id": f"changed_{len(self.changes)}",
                    "args": {
                        key: value for key, value in e2e.ARGS.items()
                        if key in other.tool_call_schema.model_json_schema()["properties"]
                    },
                }
                return e2e.ChatResult(
                    generations=[e2e.ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
                )
            return super()._generate(messages, stop, run_manager, **kwargs)

    model = SwitchToolModel(structured=True)
    graph = e2e._graph(
        tmp_path, monkeypatch, model,
        max_tool_rounds=8, max_stagnant_tool_rounds=3,
    )
    state, rating = graph.propagate("NVDA", "2026-01-09")
    assert len(model.changes) == 3
    assert len(model.tool_turns) == 9
    assert state["market_report"] == e2e.TEXT
    assert state["news_report"] == e2e.TEXT
    assert state["fundamentals_report"] == e2e.TEXT
    assert rating == "Overweight"


@pytest.mark.unit
def test_trajectory_eval_counts_blocked_calls_as_resolved(tmp_path, monkeypatch, offline):
    from tradingagents.evaluation.collector import collect_trajectory

    graph = e2e._graph(
        tmp_path, monkeypatch, e2e.LoopingModel(structured=True),
        max_tool_rounds=8, max_stagnant_tool_rounds=3,
    )
    report = collect_trajectory(
        graph, "NVDA", "2026-01-09",
        case_id="corrective-block", attempt_id="1", output_dir=tmp_path / "eval",
    )
    assert report["metrics"]["tool_calls_blocked"] == 3
    assert report["metrics"]["tool_calls_requested"] == (
        report["metrics"]["tool_calls"]
        + report["metrics"]["tool_calls_reused"]
        + report["metrics"]["tool_calls_blocked"]
    )
    assert report["rules"]["evidence_completeness"]["status"] == "pass", report
    assert report["rules"]["tool_result_linkage"]["status"] == "pass", report
