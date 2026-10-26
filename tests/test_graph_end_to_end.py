"""使用脚本化模型、不联网，执行整个 Graph 的端到端测试。

每个分析师通过数据路由器各调用其工具一次，随后执行辩论与经理节点，
最终解析并记录决策。测试用于固定节点、工具及边的连接关系，
重构时若意外遗漏某个部分就应失败。"""

from __future__ import annotations

import copy
import time

import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from tradingagents.agents import context, schemas
from tradingagents.agents.analysts import sentiment_analyst
from tradingagents.dataflows import router
from tradingagents.dataflows.vendors.yahoo import market as yahoo_market, snapshot
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph

TRADE_DATE = "2026-01-09"

# 为所有自由文本解析器提供足够信息：PM 的明确评级标签，
# 以及 Trader 的最终交易提案结尾行。
TEXT = "Report.\n\n**Rating**: Overweight\n\nFINAL TRANSACTION PROPOSAL: **BUY**"

STRUCTURED = {
    schemas.ResearchPlan: schemas.ResearchPlan(
        recommendation=schemas.PortfolioRating.OVERWEIGHT, rationale="r", strategic_actions="a"),
    schemas.TraderProposal: schemas.TraderProposal(action=schemas.TraderAction.BUY, reasoning="r"),
    # 投资论点引用了别人的评级，但最终决策仍应以 PM 自身评级为准。
    schemas.PortfolioDecision: schemas.PortfolioDecision(
        rating=schemas.PortfolioRating.OVERWEIGHT, executive_summary="s",
        investment_thesis="Street consensus rating: Buy (28 of 35 analysts)."),
    schemas.SentimentReport: schemas.SentimentReport(
        overall_band=schemas.SentimentBand.NEUTRAL, overall_score=5.0, confidence="low", narrative="n"),
}

ARGS = {"symbol": "NVDA", "ticker": "NVDA", "curr_date": TRADE_DATE, "start_date": "2026-01-02",
        "end_date": TRADE_DATE, "indicator": "rsi", "topic": "Fed rate cut", "freq": "quarterly"}


class ScriptedModel(BaseChatModel):
    """每个绑定工具调用一次，然后输出 TEXT。"""

    structured: bool = False
    tools: tuple = ()
    calls: list = Field(default_factory=list)   # shared across bound copies
    threads: set = Field(default_factory=set)   # threads that served a tool-bound call
    fail_at: int | None = None                  # raise on this call, once

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools": tuple(tools)})

    def with_structured_output(self, schema, **kwargs):
        if not self.structured:
            raise NotImplementedError
        return RunnableLambda(lambda _: self._count() or STRUCTURED[schema])

    def _count(self) -> None:
        self.calls.append(1)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("provider unavailable")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._count()
        if self.tools:
            import threading
            import time
            self.threads.add(threading.current_thread().name)
            time.sleep(0.05)   # long enough for concurrent analysts to overlap
        if self.tools and not isinstance(messages[-1], ToolMessage):
            calls = [{"name": t.name, "id": f"call_{i}",
                      "args": {k: v for k, v in ARGS.items()
                               if k in t.tool_call_schema.model_json_schema()["properties"]}}
                     for i, t in enumerate(self.tools)]
            message = AIMessage(content="", tool_calls=calls)
        else:
            message = AIMessage(content=TEXT)
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """所有提供商离线返回结果，并收集被调用的路由方法。"""
    called: set[str] = set()
    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in vendors:
            monkeypatch.setitem(vendors, vendor,
                                lambda *a, _m=method, **k: called.add(_m) or f"{_m} data")
    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=60),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000,
    })
    monkeypatch.setattr(snapshot, "load_ohlcv",
                        lambda *a, **k: called.add("ohlcv") or prices.copy())
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages", lambda *a, **k: "no posts")
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts", lambda *a, **k: "no posts")
    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: type("T", (), {"info": {"longName": "NVIDIA"}})())
    context._identity.cache_clear()
    return called


def _graph(tmp_path, monkeypatch, model, debug=False, **config):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
               memory_log_path=str(tmp_path / "log.md"), **config)
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda config, tier, **k: _Client(model))
    return trading_graph.TradingAgentsGraph(config=cfg, debug=debug)


@pytest.mark.unit
@pytest.mark.parametrize("structured", [False, True], ids=["free-text", "structured"])
def test_a_full_run_reaches_a_logged_decision(tmp_path, monkeypatch, offline, structured):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=structured))

    state, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == state["final_rating"] == "Overweight"
    for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report",
                "investment_plan", "trader_investment_plan", "final_trade_decision"):
        assert state[key].strip(), key
    tool_methods = {"get_stock_data", "get_indicators", "get_news", "get_global_news",
                    "get_macro_indicators", "get_prediction_markets", "get_fundamentals",
                    "get_balance_sheet", "get_cashflow", "get_income_statement",
                    "get_insider_transactions", "ohlcv"}
    assert offline == tool_methods
    assert [e["rating"] for e in graph.memory_log.load_entries()] == ["Overweight"]


@pytest.mark.unit
def test_an_interrupted_run_resumes_from_its_checkpoint(tmp_path, monkeypatch, offline):
    model = ScriptedModel(fail_at=12)            # past the analysts, before the decision
    graph = _graph(tmp_path, monkeypatch, model, checkpoint_enabled=True)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        graph.propagate("NVDA", TRADE_DATE)
    calls_before = len(model.calls)

    _, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == "Overweight"
    resumed_calls = len(model.calls) - calls_before
    full_run = ScriptedModel()
    _graph(tmp_path / "fresh", monkeypatch, full_run).propagate("NVDA", TRADE_DATE)
    # 恢复后的运行只应执行中断前尚未完成的调用。
    assert resumed_calls == len(full_run.calls) - (model.fail_at - 1)


@pytest.mark.unit
def test_a_graph_reused_across_runs_keeps_no_run_state(tmp_path, monkeypatch, offline):
    """回测会在整个任务网格中复用 Graph；若保存每轮完整状态，
    内存占用会随任务数无限增长。"""
    graph = _graph(tmp_path, monkeypatch, ScriptedModel())
    for trade_date in ("2026-01-08", TRADE_DATE):
        graph.propagate("NVDA", trade_date)

    held = [v for v in vars(graph).values() if isinstance(v, dict) and TRADE_DATE in v]
    assert held == []
    assert len(list(tmp_path.glob("results/NVDA/TradingAgentsStrategy_logs/*.json"))) == 2


@pytest.mark.unit
def test_the_analysts_run_at_the_same_time(tmp_path, monkeypatch, offline):
    model = ScriptedModel()
    graph = _graph(tmp_path, monkeypatch, model)

    assert not [n for n in graph.graph.get_graph().nodes if n.startswith("Msg Clear")]
    graph.propagate("NVDA", TRADE_DATE)

    assert len(model.threads) > 1


@pytest.mark.unit
def test_a_debug_run_prints_the_analysts_work_and_reaches_the_same_decision(tmp_path, monkeypatch, offline, capsys):
    """调试模式流式执行分析师各自的子图，工具调用也应显示。"""
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(), debug=True)

    state, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == "Overweight"
    assert state["market_report"].strip() and state["fundamentals_report"].strip()
    printed = capsys.readouterr().out
    assert "get_stock_data" in printed and "get_balance_sheet" in printed


@pytest.mark.unit
def test_each_report_streams_as_soon_as_its_analyst_files_it(tmp_path, monkeypatch, offline):
    """顶层状态要等所有分析师完成才合并报告；
    CLI 则应在各报告完成时立即展示，并分别停止计时。"""
    graph = _graph(tmp_path, monkeypatch, ScriptedModel())
    reports = ("market_report", "sentiment_report", "news_report", "fundamentals_report")

    first = next(state for _, state in graph.stream_run(graph.create_run_state("NVDA", TRADE_DATE),
                                                        **graph.propagator.get_graph_args())
                 if state and any(state.get(k) for k in reports))

    assert sum(bool(first.get(k)) for k in reports) == 1


@pytest.mark.unit
def test_a_streamed_run_reads_its_own_graph_config(tmp_path, monkeypatch, offline):
    """CLI 使用流式运行；其他地方修改进程全局配置后，
    不能影响本 Graph 的工具调用，Graph 的配置也不能泄漏回调用方。"""
    from tradingagents.dataflows.config import get_config, set_config

    graph = _graph(tmp_path, monkeypatch, ScriptedModel(), output_language="French")
    set_config({"output_language": "German"})
    seen = []
    monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "yfinance",
                        lambda *a, **k: seen.append(get_config()["output_language"]) or "prices")

    between = [get_config()["output_language"]
               for _ in graph.stream_run(graph.create_run_state("NVDA", TRADE_DATE), **graph.propagator.get_graph_args())]

    assert seen and set(seen) == {"French"}
    assert set(between) == {"German"}


class LoopingModel(ScriptedModel):
    """只要有工具可用，模型每轮都会调用工具（#1420）。"""

    tool_turns: list = Field(default_factory=list)    # shared across bound copies
    last_turns: list = Field(default_factory=list)    # histories of the turns offered no tools

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.tools:
            self.last_turns.append(messages)
            return super()._generate(messages, stop, run_manager, **kwargs)
        self.tool_turns.append(1)
        tool = self.tools[0]
        call = {"name": tool.name, "id": f"call_{len(self.tool_turns)}",
                "args": {k: v for k, v in ARGS.items()
                         if k in tool.tool_call_schema.model_json_schema()["properties"]}}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


@pytest.mark.unit
def test_an_analyst_that_keeps_calling_tools_writes_its_report_at_the_limit(tmp_path, monkeypatch, offline):
    """不停调用工具的模型不能阻止整个运行正常收尾（#1420）。"""
    model = LoopingModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, max_tool_rounds=3)

    final_state, rating = graph.propagate("NVDA", TRADE_DATE)

    assert len(model.tool_turns) == 3 * 3   # three tool-using analysts, three rounds each
    for key in ("market_report", "news_report", "fundamentals_report"):
        assert final_state[key] == TEXT
    assert rating == "Overweight"


@pytest.mark.unit
def test_the_last_turn_is_offered_no_tools_and_reads_its_tool_results_as_text(tmp_path, monkeypatch, offline):
    """只要工具仍绑定，模型可能忽略停止指令继续调用；
    但移除工具后，部分提供商又拒绝包含历史工具调用的消息。"""
    model = LoopingModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, max_tool_rounds=2)

    graph.propagate("NVDA", TRADE_DATE)

    wrap_ups = [h for h in model.last_turns if isinstance(h[-1], HumanMessage) and "tool round" in h[-1].content]
    assert len(wrap_ups) == 3
    for history in wrap_ups:
        assert not any(isinstance(m, ToolMessage) or getattr(m, "tool_calls", None) for m in history)
        assert any("returned]" in str(m.content) for m in history)
        assert "tool rounds are spent" in history[0].content   # the system prompt lists no tools


@pytest.mark.unit
def test_a_tool_limit_the_recursion_limit_cannot_hold_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="max_tool_rounds"):
        _graph(tmp_path, monkeypatch, ScriptedModel(), max_tool_rounds=60, max_recur_limit=100)


class TimedModel(ScriptedModel):
    """记录每次调用所处 Graph 步骤、Prompt 和时间的脚本化模型。"""

    seen: list = Field(default_factory=list)    # shared across bound copies

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        step = (run_manager.metadata or {}).get("langgraph_node") if run_manager else None
        started = time.monotonic()
        if step in {"Market Analyst", "agent"} or step == "Memory Log":
            time.sleep(0.2)                     # long enough to see what overlaps
        result = super()._generate(messages, stop, run_manager, **kwargs)
        prompt = "\n".join(str(m.content) for m in messages)
        self.seen.append((step, prompt, started, time.monotonic()))
        return result


@pytest.mark.unit
def test_past_decisions_settle_alongside_the_analysts(tmp_path, monkeypatch, offline):
    from tradingagents.memory import settlement

    model = TimedModel()
    graph = _graph(tmp_path, monkeypatch, model)
    graph.memory_log.store_decision("NVDA", "2025-12-31", "Rating: Buy\n\nBuy NVDA.")
    closes = pd.Series([100.0 + i for i in range(10)], index=pd.bdate_range("2026-01-02", periods=10))
    monkeypatch.setattr(settlement, "get_closes", lambda *a, **k: closes)

    graph.propagate("NVDA", TRADE_DATE)

    reflections = [c for c in model.seen if c[0] == "Memory Log"]
    assert len(reflections) == 1                # settled once, inside the run
    analyst_turns = [c for c in model.seen if c[0] == "agent"]
    assert reflections[0][2] < max(c[3] for c in analyst_turns), "settled before the analysts, not alongside"
    settled = next(e for e in graph.memory_log.load_entries() if e["date"] == "2025-12-31")
    assert not settled["pending"]
    lessons = graph.memory_log.get_past_context("NVDA", as_of=TRADE_DATE)
    assert lessons
    pm_prompt = next(c[1] for c in model.seen if c[0] == "Portfolio Manager")
    assert lessons in pm_prompt


@pytest.mark.unit
def test_a_settlement_failure_is_noted_and_the_run_goes_on(tmp_path, monkeypatch, offline):
    """Graph 汇合节点等待结算时，分析师已耗尽其模型调用预算。"""
    from tradingagents.reporting import write_report_tree

    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True))

    def locked():
        raise OSError("memory log is locked")

    monkeypatch.setattr(graph, "settle_all_pending", locked)
    final_state, rating = graph.propagate("NVDA", TRADE_DATE)

    assert rating == "Overweight"
    assert "could not be settled" in final_state["memory_note"]
    report = write_report_tree(final_state, "NVDA", tmp_path / "report").read_text()
    assert "Memory log: Past decisions could not be settled" in report


@pytest.mark.unit
def test_the_run_settles_other_tickers_due_decisions_too(tmp_path, monkeypatch, offline):
    """即使调度器停止分析某个 ticker，其历史决策仍需结算（#1445）。"""
    from tradingagents.memory import settlement

    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True))
    graph.memory_log.store_decision("MSFT", "2026-01-02", "Rating: Buy\n\nBuy MSFT.")
    closes = pd.Series([100.0 + i for i in range(10)], index=pd.bdate_range("2026-01-02", periods=10))
    monkeypatch.setattr(settlement, "get_closes", lambda *a, **k: closes)

    graph.propagate("NVDA", TRADE_DATE)

    msft = next(e for e in graph.memory_log.load_entries() if e["ticker"] == "MSFT")
    assert not msft["pending"]


@pytest.mark.unit
def test_the_memory_step_reads_lessons_as_of_the_trade_date_and_settles_once(tmp_path, monkeypatch, offline):
    """历史日期运行只读取当时已知的经验（#1251）；
    从检查点恢复时也不能重复结算同一决策。"""
    from tradingagents.memory import settlement

    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True))
    graph.memory_log.store_decision("NVDA", "2025-12-31", "Rating: Buy\n\nBuy NVDA.")
    closes = pd.Series([100.0 + i for i in range(10)], index=pd.bdate_range("2026-01-02", periods=10))
    monkeypatch.setattr(settlement, "get_closes", lambda *a, **k: closes)
    asked = []
    real = graph.memory_log.get_past_context
    monkeypatch.setattr(graph.memory_log, "get_past_context",
                        lambda ticker, as_of=None, **k: asked.append(as_of) or real(ticker, as_of=as_of, **k))
    state = {"company_of_interest": "NVDA", "trade_date": "2026-01-20"}

    graph._memory_step(state)
    log_after_first = (tmp_path / "log.md").read_text()
    graph._memory_step(state)

    assert asked == ["2026-01-20", "2026-01-20"]
    assert (tmp_path / "log.md").read_text() == log_after_first
