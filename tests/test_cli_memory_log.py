"""CLI 应与 propagate() 共用相同的 Memory Log 流程。

CLI 曾直接流式运行 Graph，没有执行仅在 propagate() 中的步骤：
未结算待处理决策、没有给 Portfolio Manager 注入历史经验，
也没有记录最终决策。现在两者使用相同的 Graph 方法
构造初始状态并记录决策。"""

from __future__ import annotations

import pytest

import cli.run as cli_run
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import TradingMemoryLog


def _bare_graph(tmp_path):
    """绕过 __init__ 构造 Graph，不启动 LLM 客户端，连接临时日志。"""
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_log_path": str(tmp_path / "trading_memory.md")}
    graph.memory_log = TradingMemoryLog(graph.config)
    graph._log_state = lambda *a: None   # these tests are about the memory log
    return graph


@pytest.mark.unit
def test_record_decision_appends_a_pending_entry(tmp_path):
    graph = _bare_graph(tmp_path)
    graph.record_decision("NVDA", "2026-01-10", {"final_trade_decision": "Rating: Buy\n\nBuy NVDA."})
    entries = graph.memory_log.load_entries()
    assert [(e["ticker"], e["pending"], e["rating"]) for e in entries] == [("NVDA", True, "Buy")]


@pytest.mark.unit
def test_record_decision_skips_a_run_without_a_decision(tmp_path):
    graph = _bare_graph(tmp_path)
    graph.record_decision("NVDA", "2026-01-10", {})
    assert graph.memory_log.load_entries() == []


# --- CLI 运行流程 ----------------------------------------------------------------

class _FakeGraph:
    """记录 run_analysis 触发的生命周期方法。"""

    def __init__(self, resuming=None):
        self.calls = []
        self.graph = self
        self.propagator = self
        self.resuming = resuming      # None: checkpointing off
        self._resuming = False

    def create_run_state(self, ticker, trade_date, asset_type="stock", portfolio=None):
        self.calls.append(("create_run_state", ticker, trade_date))
        return {"messages": [], "company_of_interest": ticker}

    def record_decision(self, ticker, trade_date, final_state):
        self.calls.append(("record_decision", ticker, trade_date, final_state.get("final_trade_decision")))

    def get_graph_args(self, callbacks=None):
        return {}

    def begin_checkpoint(self, *a, **k):
        self._resuming = bool(self.resuming)
        return None if self.resuming is None else "thread"

    def checkpoint_input(self, state):
        return state

    def clear_checkpoint_on_success(self, *a, **k):
        self.calls.append(("clear_checkpoint",))

    def end_checkpoint(self):
        pass

    def stream_run(self, graph_input, **kwargs):
        yield [], {"messages": [], "market_report": "M"}
        yield [], {"messages": [], "final_trade_decision": "Rating: Buy\n\nBuy NVDA.", "final_rating": "Buy"}


class _NullLive:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeBuffer:
    def __init__(self):
        self.messages = []
        self.tool_calls = []
        self.report_sections = {}
        self.agent_status = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        self.selected_analysts = [a.lower() for a in selected_analysts]

    def add_message(self, kind, content):
        self.messages.append((0.0, kind, content))

    def add_tool_call(self, name, args):
        self.tool_calls.append((0.0, name, args))

    def update_report_section(self, *a):
        pass

    def update_agent_status(self, agent, status):
        self.agent_status[agent] = status


def _run_cli(monkeypatch, tmp_path, fake):
    """使用 ``fake`` 执行 run_analysis，并返回消息缓冲区。"""
    import cli.main as m
    from cli.models import AnalystType

    buffer = _FakeBuffer()
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", lambda *a, **k: fake)
    monkeypatch.setattr(cli_run, "message_buffer", buffer)
    monkeypatch.setattr(cli_run, "create_layout", lambda: None)
    monkeypatch.setattr(cli_run, "update_display", lambda *a, **k: None)
    monkeypatch.setattr(cli_run, "Live", _NullLive)
    monkeypatch.setattr(cli_run, "get_user_selections", lambda flags=None: {
        "ticker": "NVDA", "analysis_date": "2026-01-10",
        "analysts": [AnalystType.MARKET], "asset_type": "stock",
    })
    monkeypatch.setattr(cli_run, "_build_run_config", lambda selections, checkpoint: {
        "data_cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "results"),
    })
    monkeypatch.setattr(m.typer, "prompt", lambda *a, **k: "N")
    cli_run.run_analysis()
    return buffer


@pytest.mark.unit
def test_cli_run_uses_the_memory_log_like_propagate(tmp_path, monkeypatch):
    fake = _FakeGraph()
    _run_cli(monkeypatch, tmp_path, fake)

    assert fake.calls == [
        ("create_run_state", "NVDA", "2026-01-10"),
        # 从合并后的状态流中记录最终决策，且必须在清除检查点之前完成；
        # 这与 propagate() 的行为相同。
        ("record_decision", "NVDA", "2026-01-10", "Rating: Buy\n\nBuy NVDA."),
        ("clear_checkpoint",),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("resuming, said", [(True, "resuming"), (False, "starting fresh")])
def test_the_cli_run_says_whether_it_resumed(tmp_path, monkeypatch, resuming, said):
    """README 要求运行界面明确区分恢复执行和全新启动。"""
    buffer = _run_cli(monkeypatch, tmp_path, _FakeGraph(resuming=resuming))

    assert any(said in text.lower() for _, kind, text in buffer.messages if kind == "System")


@pytest.mark.unit
def test_a_run_without_checkpointing_says_nothing_about_resuming(tmp_path, monkeypatch):
    buffer = _run_cli(monkeypatch, tmp_path, _FakeGraph())

    assert not any("resum" in text.lower() or "fresh" in text.lower() for _, _, text in buffer.messages)


@pytest.mark.unit
def test_recording_a_run_writes_its_state_log(tmp_path):
    """CLI 通过 record_decision 保存决策，因此状态日志在这里写入。"""
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"results_dir": str(tmp_path), "llm_provider": "openai", "deep_think_llm": "d",
                    "quick_think_llm": "q", "max_debate_rounds": 1, "max_risk_discuss_rounds": 1,
                    "output_language": "English", "data_vendors": {}, "tool_vendors": {}}
    graph.selected_analysts = ("market",)
    graph.memory_log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    state = {"company_of_interest": "NVDA", "trade_date": "2026-09-23", "market_report": "M",
             "sentiment_report": "", "news_report": "", "fundamentals_report": "",
             "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "",
                                         "current_response": ""},
             "trader_investment_plan": "T", "investment_plan": "P",
             "risk_debate_state": {"aggressive_history": "", "conservative_history": "",
                                   "neutral_history": "", "history": ""},
             "final_trade_decision": "**Rating**: Hold", "final_rating": "Hold"}

    graph.record_decision("NVDA", "2026-09-23", state)

    assert list(tmp_path.glob("NVDA/TradingAgentsStrategy_logs/full_states_log_2026-09-23.json"))
