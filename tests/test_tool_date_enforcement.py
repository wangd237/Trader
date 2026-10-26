"""带日期参数的工具必须从 Graph 状态读取分析日期，而不能信任模型输入。

各数据源的时点保护依赖接收到的日期；
模型如果漏传日期或填入今天，就可能绕过历史约束。
运行日期必须从状态注入，并在模型可见的工具 Schema 中隐藏。"""

from __future__ import annotations

from unittest import mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import tools
from tradingagents.dataflows.date_window import as_of, as_of_window

TRADE_DATE = "2026-08-14"


@pytest.mark.unit
@pytest.mark.parametrize("requested, expected", [
    ("2026-09-14", TRADE_DATE),     # later than the run: clamped
    ("2026-08-01", "2026-08-01"),   # earlier: narrows, allowed
    (None, TRADE_DATE),             # omitted
    ("Sept 1", TRADE_DATE),         # unparseable
    ("", TRADE_DATE),
])
def test_as_of_takes_the_earlier_date(requested, expected):
    assert as_of(requested, TRADE_DATE) == expected


@pytest.mark.unit
def test_as_of_without_a_trade_date_passes_the_request_through():
    assert as_of("2026-09-14", "") == "2026-09-14"


@pytest.mark.unit
@pytest.mark.parametrize("start, end, expected", [
    ("2026-08-01", "2026-09-14", ("2026-08-01", TRADE_DATE)),  # end clamped
    ("2026-08-01", "2026-08-10", ("2026-08-01", "2026-08-10")),  # inside: unchanged
    ("2026-09-01", "2026-09-08", ("2026-08-07", TRADE_DATE)),  # wholly later: span kept, moved back
])
def test_as_of_window(start, end, expected):
    assert as_of_window(start, end, TRADE_DATE) == expected


DATED_TOOLS = [
    tools.get_stock_data,
    tools.get_fundamentals,
    tools.get_balance_sheet,
    tools.get_cashflow,
    tools.get_income_statement,
    tools.get_news,
    tools.get_global_news,
    tools.get_indicators,
    tools.get_macro_indicators,
    tools.get_verified_market_snapshot,
]


@pytest.mark.unit
@pytest.mark.parametrize("tool", DATED_TOOLS, ids=lambda t: t.name)
def test_trade_date_is_hidden_from_the_model(tool):
    assert "trade_date" not in tool.tool_call_schema.model_json_schema()["properties"]


INSTRUMENT_TOOLS = [
    tools.get_stock_data, tools.get_indicators, tools.get_verified_market_snapshot,
    tools.get_fundamentals, tools.get_balance_sheet, tools.get_cashflow,
    tools.get_income_statement, tools.get_news, tools.get_insider_transactions,
]


@pytest.mark.unit
@pytest.mark.parametrize("tool", INSTRUMENT_TOOLS, ids=lambda t: t.name)
def test_the_instrument_is_hidden_from_the_model(tool):
    """模型曾将指标名 rsi、atr 误当 ticker（它们也可能是合法代码），
    导致把其他公司的数据插入报告。"""
    properties = tool.tool_call_schema.model_json_schema()["properties"]
    assert "symbol" not in properties and "ticker" not in properties


@pytest.mark.unit
def test_a_tool_serves_the_run_instrument_whatever_the_model_passes():
    args = _run(tools.get_indicators, {"symbol": "RSI", "indicator": "rsi", "curr_date": TRADE_DATE}, tools)
    assert args[1] == "NVDA"


class _State(MessagesState):
    trade_date: str
    company_of_interest: str


def _run(tool, args, module):
    """通过带有 trade_date 状态的 Graph 中 ToolNode 调用工具。"""
    graph = StateGraph(_State)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    with mock.patch.object(module, "route_to_vendor", return_value="ok") as routed:
        graph.compile().invoke({
            "messages": [AIMessage("", tool_calls=[{"name": tool.name, "args": args, "id": "1"}])],
            "trade_date": TRADE_DATE,
            "company_of_interest": "NVDA",
        })
    return routed.call_args.args


@pytest.mark.unit
def test_statement_tool_with_omitted_date_uses_the_run_date():
    args = _run(tools.get_balance_sheet, {}, tools)
    assert args[-1] == TRADE_DATE  # #1331: an omitted date no longer means unfiltered


@pytest.mark.unit
def test_future_curr_date_from_the_model_is_clamped():
    args = _run(tools.get_fundamentals, {"curr_date": "2026-09-14"}, tools)
    assert args == ("get_fundamentals", "NVDA", TRADE_DATE)


@pytest.mark.unit
def test_future_window_from_the_model_is_clamped():
    args = _run(tools.get_stock_data, {"start_date": "2026-08-01", "end_date": "2026-09-14"}, tools)
    assert args == ("get_stock_data", "NVDA", "2026-08-01", TRADE_DATE)


@pytest.mark.unit
def test_direct_call_without_state_is_unchanged():
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        tools.get_news.func("AAPL", "2026-09-01", "2026-09-08")
    assert routed.call_args.args == ("get_news", "AAPL", "2026-09-01", "2026-09-08")


@pytest.mark.unit
@pytest.mark.parametrize("tool", ["get_news", "get_stock_data"])
def test_a_start_date_that_is_not_a_date_is_sent_back_to_the_model(tool):
    # 模型可能在日期后多输出无效字符（#1476）。
    # 不能猜测窗口，应跳过提供商调用并提示模型正确格式。
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        out = getattr(tools, tool).func("NVDA", "2026-09-26\u0629", "2026-10-03", trade_date="2026-10-03")
    assert not routed.called
    assert "YYYY-MM-DD" in out and "start_date" in out


@pytest.mark.unit
def test_a_window_whose_start_is_not_a_date_is_refused():
    with pytest.raises(ValueError, match="start_date"):
        as_of_window("2026-09-26\u0629", "2026-10-03", "2026-10-03")


# --- the run date itself (#1319) -------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("bad", ["2026-9-10", "2026-09-10 00:00", "Sept 10", None])
def test_propagate_rejects_a_non_canonical_date(bad):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        object.__new__(TradingAgentsGraph).propagate("AAPL", bad)


@pytest.mark.unit
def test_propagate_rejects_a_future_date(monkeypatch):
    import tradingagents.graph.trading_graph as tg

    monkeypatch.setattr(tg, "get_current_date", lambda: "2026-09-10")
    with pytest.raises(ValueError, match="future"):
        object.__new__(tg.TradingAgentsGraph).propagate("AAPL", "2026-09-11")
