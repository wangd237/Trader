"""Offline regression tests for the conservative runtime tool-loop guard."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.test_graph_end_to_end import offline
from tradingagents.graph.tool_loop_guard import consecutive_identical_tool_rounds


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
    assert len(model.tool_turns) == 9  # 3 tool-using analysts * 3 identical rounds
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
