"""TradingMemoryLog 回归测试：存储、延迟反思、向投资组合经理注入经验，以及移除旧机制。"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import TradingMemoryLog, settlement
from tradingagents.memory.reflection import Reflector

_SEP = TradingMemoryLog._SEPARATOR

DECISION_BUY = "Rating: Buy\nEnter at $189-192, 6% portfolio cap."
DECISION_OVERWEIGHT = (
    "Rating: Overweight\n"
    "Executive Summary: Moderate position, await confirmation.\n"
    "Investment Thesis: Strong fundamentals but near-term headwinds."
)
DECISION_SELL = "Rating: Sell\nExit position immediately."
DECISION_NO_RATING = (
    "Executive Summary: Complex situation with multiple competing factors.\n"
    "Investment Thesis: No clear directional signal at this time."
)


# ---------------------------------------------------------------------------
# 共用辅助函数
# ---------------------------------------------------------------------------

def make_log(tmp_path, filename="trading_memory.md"):
    config = {"memory_log_path": str(tmp_path / filename)}
    return TradingMemoryLog(config)


def _seed_completed(tmp_path, ticker, date, decision_text, reflection_text, filename="trading_memory.md"):
    """绕过 API，直接向文件写入一条已完成记录。"""
    entry = (
        f"[{date} | {ticker} | Buy | +1.0% | +0.5% | 5d]\n\n"
        f"DECISION:\n{decision_text}\n\n"
        f"REFLECTION:\n{reflection_text}"
        + _SEP
    )
    with open(tmp_path / filename, "a", encoding="utf-8") as f:
        f.write(entry)


def _resolve_entry(log, ticker, date, decision, reflection="Good call."):
    """先保存决策，再通过 API 立即结算。"""
    log.store_decision(ticker, date, decision)
    log.update_with_outcome(ticker, date, 0.05, 0.02, 5, reflection)


def _price_df(prices, start="2026-01-05"):
    """构造与 yfinance .history() 输出一致的最小 DataFrame。

    使用真实行情同样的 DatetimeIndex，保证结算日期提取
    （stock.index[holding_days]）能够工作（#1251）。"""
    idx = pd.date_range(start=start, periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def _make_pm_state(past_context=""):
    """构造供 portfolio_manager_node 使用的最小 AgentState 字典。"""
    return {
        "company_of_interest": "NVDA",
        "past_context": past_context,
        "risk_debate_state": {
            "history": "Risk debate history.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_plan": "Research plan.",
        "trader_investment_plan": "Trader plan.",
    }


def _structured_pm_llm(captured: dict, decision: PortfolioDecision | None = None):
    """构造 MagicMock LLM：结构化绑定时捕获 Prompt，
    返回真实 PortfolioDecision，使 render_pm_decision 可以执行。"""
    if decision is None:
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold the position; await catalyst.",
            investment_thesis="Balanced view; neither side carried the debate.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# 核心：存储和读取
# ---------------------------------------------------------------------------

class TestTradingMemoryLogCore:

    def test_store_creates_file(self, tmp_path):
        log = make_log(tmp_path)
        assert not (tmp_path / "trading_memory.md").exists()
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert (tmp_path / "trading_memory.md").exists()

    def test_store_appends_not_overwrites(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        entries = log.load_entries()
        assert len(entries) == 2
        assert entries[0]["ticker"] == "NVDA"
        assert entries[1]["ticker"] == "AAPL"

    def test_store_decision_idempotent(self, tmp_path):
        """对相同 (ticker, date) 重复调用 store_decision，只能保存一条。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert len(log.load_entries()) == 1

    def test_store_decision_idempotent_after_the_entry_resolves(self, tmp_path):
        """已结算记录也必须阻止重复写入。

        旧幂等判断只检查待结算记录；同一标的、同一日期结算完成后
        再次运行，会把相同决策重新追加，导致历史经验和汇总统计重复计算。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.05, 0.02, 5, "worked", "2026-01-17")

        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)

        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["pending"] is False  # the settled record is kept, not replaced

    def test_batch_update_resolves_multiple_entries(self, tmp_path):
        """batch_update_with_outcomes 在一次写入中结算多条待处理记录。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.store_decision("NVDA", "2026-01-12", DECISION_SELL)

        updates = [
            {"ticker": "NVDA", "trade_date": "2026-01-05",
             "raw_return": 0.05, "alpha_return": 0.02, "holding_days": 5,
             "reflection": "First correct."},
            {"ticker": "NVDA", "trade_date": "2026-01-12",
             "raw_return": -0.03, "alpha_return": -0.01, "holding_days": 5,
             "reflection": "Second correct."},
        ]
        log.batch_update_with_outcomes(updates)

        entries = log.load_entries()
        assert len(entries) == 2
        assert all(not e["pending"] for e in entries)
        assert entries[0]["reflection"] == "First correct."
        assert entries[1]["reflection"] == "Second correct."

    def test_pending_tag_format(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | pending]" in text

    # 评级解析

    def test_rating_parsed_buy(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries()[0]["rating"] == "Buy"

    def test_rating_parsed_overweight(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        assert log.load_entries()[0]["rating"] == "Overweight"

    def test_an_unreadable_decision_is_tagged_for_review(self, tmp_path):
        """不能默认 Hold：虚构评级会被下一次运行当作真实历史选择，
        并进入回测统计。"""
        from tradingagents.agents.rating import RATING_REVIEW

        log = make_log(tmp_path)
        log.store_decision("MSFT", "2026-01-12", DECISION_NO_RATING)
        assert log.load_entries()[0]["rating"] == RATING_REVIEW

    def test_rating_priority_over_prose(self, tmp_path):
        """即使正文前面出现相反评级词，也应以明确的 'Rating: X' 标签为准。"""
        decision = (
            "The sell thesis is weak. The hold case is marginal.\n\n"
            "Rating: Buy\n\n"
            "Executive Summary: Strong fundamentals support the position."
        )
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"

    # 记录分隔符健壮性

    def test_decision_with_markdown_separator(self, tmp_path):
        """LLM 决策文本包含 '---' 时不能破坏记录分隔结构。"""
        decision = "Rating: Buy\n\n---\n\nRisk: elevated volatility."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        entries = log.load_entries()
        assert len(entries) == 1
        assert "Risk: elevated volatility" in entries[0]["decision"]

    # 加载日志记录

    def test_load_entries_empty_file(self, tmp_path):
        log = make_log(tmp_path)
        assert log.load_entries() == []

    def test_load_entries_single(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["date"] == "2026-01-10"
        assert e["ticker"] == "NVDA"
        assert e["rating"] == "Buy"
        assert e["pending"] is True
        assert e["raw"] is None

    def test_load_entries_multiple(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", DECISION_OVERWEIGHT)
        log.store_decision("MSFT", "2026-01-12", DECISION_NO_RATING)
        entries = log.load_entries()
        assert len(entries) == 3
        assert [e["ticker"] for e in entries] == ["NVDA", "AAPL", "MSFT"]

    def test_decision_content_preserved(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries()[0]["decision"] == DECISION_BUY.strip()

    # 获取待结算记录

    def test_get_pending_returns_pending_only(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "NVDA", "2026-01-05", "Buy NVDA.", "Correct.")
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        pending = log.get_pending_entries()
        assert len(pending) == 1
        assert pending[0]["ticker"] == "NVDA"
        assert pending[0]["date"] == "2026-01-10"

    # 读取历史经验上下文

    def test_get_past_context_empty(self, tmp_path):
        log = make_log(tmp_path)
        assert log.get_past_context("NVDA") == ""

    def test_get_past_context_pending_excluded(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.get_past_context("NVDA") == ""

    def test_get_past_context_same_ticker(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "NVDA", "2026-01-05", "Buy NVDA — AI capex thesis intact.", "Directionally correct.")
        ctx = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in ctx
        assert "Buy NVDA" in ctx

    def test_get_past_context_cross_ticker(self, tmp_path):
        log = make_log(tmp_path)
        _seed_completed(tmp_path, "AAPL", "2026-01-05", "Buy AAPL — Services growth.", "Correct.")
        ctx = log.get_past_context("NVDA")
        assert "Recent cross-ticker lessons" in ctx
        assert "Past analyses of NVDA" not in ctx

    def test_n_same_limit_respected(self, tmp_path):
        """相同 ticker 只注入最近 n_same 条经验。"""
        log = make_log(tmp_path)
        for i in range(6):
            _seed_completed(tmp_path, "NVDA", f"2026-01-{i+1:02d}", f"Buy entry {i}.", "Correct.")
        ctx = log.get_past_context("NVDA", n_same=5)
        assert "Buy entry 0" not in ctx
        assert "Buy entry 5" in ctx

    def test_n_cross_limit_respected(self, tmp_path):
        """其他 ticker 只注入最近 n_cross 条经验。"""
        log = make_log(tmp_path)
        for i, ticker in enumerate(["AAPL", "MSFT", "GOOG", "META"]):
            _seed_completed(tmp_path, ticker, f"2026-01-{i+1:02d}", f"Buy {ticker}.", "Correct.")
        ctx = log.get_past_context("NVDA", n_cross=3)
        assert "AAPL" not in ctx
        assert "META" in ctx

    # 配置为 None 时不做任何处理

    def test_no_log_path_is_noop(self):
        log = TradingMemoryLog(config=None)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        assert log.load_entries() == []
        assert log.get_past_context("NVDA") == ""

    # 轮转清理：已结算记录数量上限为可选配置

    def test_rotation_disabled_by_default(self, tmp_path):
        """没有设置 max_entries 时保留全部已结算记录。"""
        log = make_log(tmp_path)
        for i in range(7):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        assert len(log.load_entries()) == 7

    def test_rotation_prunes_oldest_resolved(self, tmp_path):
        """设置 max_entries 并超过上限时删除最早的已结算记录。"""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 3,
        })
        # 结算五条记录后，轮转应保留最近三条。
        for i in range(5):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        entries = log.load_entries()
        assert len(entries) == 3
        # 确认删除的是最旧记录，而非最新记录。
        dates = [e["date"] for e in entries]
        assert dates == ["2026-01-03", "2026-01-04", "2026-01-05"]

    def test_rotation_never_prunes_pending(self, tmp_path):
        """待结算记录不受数量上限影响，必须始终保留。"""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 2,
        })
        # 三条已结算、两条待结算：上限为二时仅保留两条已结算，待结算全部保留。
        for i in range(3):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Resolved {i}.")
        log.store_decision("NVDA", "2026-02-01", DECISION_BUY)
        log.store_decision("NVDA", "2026-02-02", DECISION_OVERWEIGHT)
        # 再结算一条以触发轮转，待结算记录仍必须保留。
        _resolve_entry(log, "NVDA", "2026-01-04", DECISION_BUY, "Resolved 3.")
        entries = log.load_entries()
        pending = [e for e in entries if e["pending"]]
        resolved = [e for e in entries if not e["pending"]]
        assert len(pending) == 2, "pending entries must never be pruned"
        assert len(resolved) == 2, f"expected 2 resolved after rotation, got {len(resolved)}"

    def test_rotation_under_cap_is_noop(self, tmp_path):
        """已结算记录数不超过 max_entries 时不执行轮转清理。"""
        log = TradingMemoryLog({
            "memory_log_path": str(tmp_path / "trading_memory.md"),
            "memory_log_max_entries": 10,
        })
        for i in range(3):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        assert len(log.load_entries()) == 3

    # 评级解析：Markdown 加粗与编号列表格式

    def test_rating_parsed_from_bold_markdown(self, tmp_path):
        """**Rating**: Buy 中标签的 Markdown 加粗不能妨碍解析。"""
        decision = "**Rating**: Buy\nEnter at $190."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"

    def test_rating_parsed_from_bold_value(self, tmp_path):
        """Rating: **Sell** 中评级值加粗也必须可解析。"""
        decision = "Rating: **Sell**\nExit immediately."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Sell"

    def test_rating_label_wins_over_prose_with_markdown(self, tmp_path):
        """即使正文出现冲突评级词，明确的 Rating: **Sell** 仍应优先。"""
        decision = (
            "The buy thesis is weakened by guidance.\n"
            "Rating: **Sell**\n"
            "Exit before earnings."
        )
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Sell"

    def test_rating_parsed_from_numbered_list(self, tmp_path):
        """带编号前缀的 1. Rating: Buy 也必须可解析。"""
        decision = "1. Rating: Buy\nEnter at $190."
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", decision)
        assert log.load_entries()[0]["rating"] == "Buy"


# ---------------------------------------------------------------------------
# 延迟反思：结果写入、反思与收益计算
# ---------------------------------------------------------------------------

class TestDeferredReflection:

    # 写入单条决策结果

    def test_update_replaces_pending_tag(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | pending]" not in text
        assert "+4.2%" in text
        assert "+2.1%" in text
        assert "5d" in text

    def test_update_appends_reflection(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["pending"] is False
        assert e["reflection"] == "Momentum confirmed."
        assert e["decision"] == DECISION_BUY.strip()

    def test_update_preserves_other_entries(self, tmp_path):
        """只修改匹配的记录，其余记录保持原样。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.store_decision("AAPL", "2026-01-11", "Rating: Hold\nHold AAPL.")
        log.store_decision("MSFT", "2026-01-12", DECISION_SELL)
        log.update_with_outcome("AAPL", "2026-01-11", 0.01, -0.01, 5, "Neutral result.")
        entries = log.load_entries()
        assert len(entries) == 3
        nvda, aapl, msft = entries
        assert nvda["ticker"] == "NVDA" and nvda["pending"] is True
        assert aapl["ticker"] == "AAPL" and aapl["pending"] is False
        assert aapl["reflection"] == "Neutral result."
        assert msft["ticker"] == "MSFT" and msft["pending"] is True

    def test_update_atomic_write(self, tmp_path):
        """存在旧 .tmp 文件时可覆盖它并正确更新日志。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        stale_tmp = tmp_path / "trading_memory.tmp"
        stale_tmp.write_text("GARBAGE CONTENT — should be overwritten", encoding="utf-8")
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Correct.")
        assert not stale_tmp.exists()
        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["reflection"] == "Correct."
        assert entries[0]["pending"] is False

    def test_update_noop_when_no_log_path(self):
        log = TradingMemoryLog(config=None)
        log.update_with_outcome("NVDA", "2026-01-10", 0.05, 0.02, 5, "Reflection")

    def test_formatting_roundtrip_after_update(self, tmp_path):
        """更新后所有字段保持完整，标记行与 DECISION 之间的空行也保留。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-10", 0.042, 0.021, 5, "Momentum confirmed.")
        entries = log.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["pending"] is False
        assert e["decision"] == DECISION_BUY.strip()
        assert e["reflection"] == "Momentum confirmed."
        assert e["raw"] == "+4.2%"
        assert e["alpha"] == "+2.1%"
        assert e["holding"] == "5d"
        raw_text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "[2026-01-10 | NVDA | Buy | +4.2% | +2.1% | 5d]\n\nDECISION:" in raw_text

    # 调用最终决策反思

    def test_reflect_on_final_decision_returns_llm_output(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Directionally correct. Thesis confirmed."
        reflector = Reflector(mock_llm)
        result = reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY, raw_return=0.042, alpha_return=0.021
        )
        assert result == "Directionally correct. Thesis confirmed."
        mock_llm.invoke.assert_called_once()

    def test_reflect_on_final_decision_includes_returns_in_prompt(self):
        """发送给 LLM 的用户消息必须包含实际收益数字。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Incorrect call."
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_SELL, raw_return=-0.08, alpha_return=-0.05
        )
        messages = mock_llm.invoke.call_args[0][0]
        human_content = next(content for role, content in messages if role == "human")
        assert "-8.0%" in human_content
        assert "-5.0%" in human_content
        assert "Exit position immediately." in human_content

    # 获取持仓窗口收益

    def test_fetch_returns_valid_ticker(self):
        stock_prices = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0, 107.0]
        spy_prices   = [400.0, 402.0, 404.0, 403.0, 405.0, 406.0, 407.0]
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-01-05")
        assert raw is not None and alpha is not None and days is not None
        assert isinstance(raw, float) and isinstance(alpha, float) and isinstance(days, int)
        assert days == 5
        # 结算日期为入场后第 `days` 个交易时段的收盘日期（#1251）。
        assert resolved == "2026-01-11"

    def test_fetch_returns_too_recent(self):
        """只有一个数据点时返回全 None，不能崩溃。"""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            m = MagicMock()
            m.history.return_value = _price_df([100.0])
            mock_ticker_cls.return_value = m
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-04-19")
        assert (raw, alpha, days, resolved) == (None, None, None, None)

    def test_fetch_returns_delisted(self):
        """空 DataFrame 返回全 None，不能崩溃。"""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            m = MagicMock()
            m.history.return_value = pd.DataFrame({"Close": []})
            mock_ticker_cls.return_value = m
            raw, alpha, days, resolved = settlement.fetch_returns("XXXXXFAKE", "2026-01-10")
        assert (raw, alpha, days, resolved) == (None, None, None, None)

    def test_fetch_returns_spy_shorter_than_stock(self):
        """SPY 数据行少于股票，但仍覆盖完整观察窗口时，
        不能发生 IndexError。"""
        stock_prices = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0, 107.0, 108.0]  # 8 rows
        spy_prices   = [400.0, 402.0, 403.0, 405.0, 406.0, 407.0, 408.0]         # 7 rows
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-01-05")
        assert raw is not None and alpha is not None
        assert days == 5  # full holding window used for both series
        assert resolved == "2026-01-11"

    def test_a_decision_is_scored_from_the_first_close_after_its_date(self):
        """#1518：分析日收盘后发布的新闻也会进入分析师上下文，
        因此不能假定用户能够按当日收盘价成交。"""
        stock_prices = [100.0, 150.0, 150.0, 150.0, 150.0, 150.0, 165.0]   # jumps on the next close
        spy_prices   = [400.0] * 7
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-01-05")
        assert raw == pytest.approx(0.10) and alpha == pytest.approx(0.10)   # 150 -> 165, not 100 -> 150
        assert resolved == "2026-01-11"

    def test_fetch_returns_incomplete_window_stays_pending(self):
        """#1169：完整持仓窗口尚未交易结束时重新运行，
        应返回全 None 并保持待结算，不能提前使用部分收益率。"""
        stock_prices = [100.0, 102.0, 104.0]  # only 3 rows; holding window is 5
        spy_prices   = [400.0, 402.0, 404.0]
        with patch("yfinance.Ticker") as mock_ticker_cls:
            def _make_ticker(sym):
                m = MagicMock()
                m.history.return_value = _price_df(spy_prices if sym == "SPY" else stock_prices)
                return m
            mock_ticker_cls.side_effect = _make_ticker
            result = settlement.fetch_returns("NVDA", "2026-01-05")
        assert result == (None, None, None, None)

    # settlement.resolve_benchmark 为 Alpha 计算选择基准指数。

    def test_resolve_benchmark_explicit_override(self):
        """config['benchmark_ticker'] 的显式配置对全部 ticker 优先。"""
        config = {
            "benchmark_ticker": "QQQ",
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert settlement.resolve_benchmark("7203.T", config) == "QQQ"
        assert settlement.resolve_benchmark("NVDA", config) == "QQQ"

    def test_resolve_benchmark_suffix_map(self):
        """已知交易所后缀对应各自的地区基准指数。"""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {
                ".T": "^N225", ".HK": "^HSI", ".NS": "^NSEI",
                ".L": "^FTSE", ".TO": "^GSPTSE", ".AX": "^AXJO",
                ".BO": "^BSESN", "": "SPY",
            },
        }
        assert settlement.resolve_benchmark("7203.T", config) == "^N225"
        assert settlement.resolve_benchmark("0700.HK", config) == "^HSI"
        assert settlement.resolve_benchmark("RELIANCE.NS", config) == "^NSEI"
        assert settlement.resolve_benchmark("AZN.L", config) == "^FTSE"

    def test_explicit_benchmark_is_resolved_like_any_other_symbol(self):
        """自定义基准也需与 ticker 使用相同别名映射，
        否则收益查询可能找不到价格，使决策无法结算。"""
        config = {"benchmark_ticker": "SPX500", "benchmark_map": {"": "SPY"}}
        assert settlement.resolve_benchmark("NVDA", config) == "^GSPC"

    def test_resolve_benchmark_china_a_shares(self):
        """A 股 ticker 应映射到所在交易所综合指数；
        测试使用真实 benchmark_map 默认配置。"""
        from tradingagents.default_config import DEFAULT_CONFIG
        config = {"benchmark_ticker": None,
                             "benchmark_map": DEFAULT_CONFIG["benchmark_map"]}
        assert settlement.resolve_benchmark("600519.SS", config) == "000001.SS"
        assert settlement.resolve_benchmark("000001.SZ", config) == "399001.SZ"
        # 交易所自身使用 .SH，而 Yahoo 的上海代码后缀是 .SS（#1260）。
        assert settlement.resolve_benchmark("600519.SH", config) == "000001.SS"

    def test_resolve_benchmark_brazil(self):
        """B3 市场标的不应再错误地对标 SPY。"""
        from tradingagents.default_config import DEFAULT_CONFIG
        config = {"benchmark_ticker": None,
                             "benchmark_map": DEFAULT_CONFIG["benchmark_map"]}
        assert settlement.resolve_benchmark("PETR4.SA", config) == "^BVSP"

    @pytest.mark.parametrize(("ticker", "index"), [
        ("2330.TW", "^TWII"), ("6488.TWO", "^TWII"), ("005930.KS", "^KS11"), ("247540.KQ", "^KQ11"), ("D05.SI", "^STI"), ("SAP.DE", "^GDAXI"),
        ("MC.PA", "^FCHI"), ("ASML.AS", "^AEX"), ("NESN.SW", "^SSMI"), ("ENI.MI", "FTSEMIB.MI"),
    ])
    def test_resolve_benchmark_regional_indexes(self, ticker, index):
        """台湾、韩国、新加坡和主要欧洲交易所不应统一错误地使用 SPY。"""
        from tradingagents.default_config import DEFAULT_CONFIG
        config = {"benchmark_ticker": None, "benchmark_map": DEFAULT_CONFIG["benchmark_map"]}
        assert settlement.resolve_benchmark(ticker, config) == index

    def test_resolve_benchmark_us_ticker_defaults_to_spy(self):
        """无点号交易所后缀的美股使用空后缀默认基准。"""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert settlement.resolve_benchmark("NVDA", config) == "SPY"
        assert settlement.resolve_benchmark("AAPL", config) == "SPY"

    def test_resolve_benchmark_unknown_suffix_falls_back(self):
        """无法识别的后缀（如 BRK.B、FAKE.XX）回退到 SPY。"""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {"": "SPY", ".T": "^N225"},
        }
        assert settlement.resolve_benchmark("FAKE.XX", config) == "SPY"
        assert settlement.resolve_benchmark("BRK.B", config) == "SPY"

    def test_resolve_benchmark_case_insensitive(self):
        """后缀匹配不区分大小写，7203.t 与 7203.T 应一致。"""
        config = {
            "benchmark_ticker": None,
            "benchmark_map": {".T": "^N225", "": "SPY"},
        }
        assert settlement.resolve_benchmark("7203.t", config) == "^N225"

    def test_reflector_includes_benchmark_in_label(self):
        """Prompt 标签应显示 benchmark_name，而不是写死 SPY。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Directionally correct."
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY,
            raw_return=0.05,
            alpha_return=0.02,
            benchmark_name="^N225",
        )
        messages = mock_llm.invoke.call_args[0][0]
        human_content = next(content for role, content in messages if role == "human")
        assert "Alpha vs ^N225:" in human_content
        assert "Alpha vs SPY:" not in human_content

    def test_reflector_defaults_to_spy_for_unupdated_callers(self):
        """未指定 benchmark_name 时为旧调用者保持 SPY 默认标签。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "ok"
        reflector = Reflector(mock_llm)
        reflector.reflect_on_final_decision(
            final_decision=DECISION_BUY,
            raw_return=0.05,
            alpha_return=0.02,
        )
        messages = mock_llm.invoke.call_args[0][0]
        human_content = next(content for role, content in messages if role == "human")
        assert "Alpha vs SPY:" in human_content

    # 结算待处理决策

    def test_resolve_skips_other_tickers(self, tmp_path):
        """分析 NVDA 时不应结算待处理的 AAPL 记录。"""
        log = make_log(tmp_path)
        log.store_decision("AAPL", "2026-01-10", DECISION_BUY)
        with patch.object(settlement, "fetch_returns", return_value=(0.05, 0.02, 5, "2026-01-12")) as fetch:
            settlement.settle_pending("NVDA", log, MagicMock(), {})
        fetch.assert_not_called()
        assert len(log.get_pending_entries()) == 1

    def test_resolve_marks_entry_completed(self, tmp_path):
        """成功结算后 get_pending_entries() 为空，记录包含 REFLECTION。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        reflector = MagicMock()
        reflector.reflect_on_final_decision.return_value = "Momentum confirmed."
        with patch.object(settlement, "fetch_returns", return_value=(0.05, 0.02, 5, "2026-01-12")):
            settlement.settle_pending("NVDA", log, reflector, {})
        assert log.get_pending_entries() == []
        entries = log.load_entries()
        assert len(entries) == 1
        assert entries[0]["pending"] is False
        assert entries[0]["reflection"] == "Momentum confirmed."
        assert "+5.0%" in entries[0]["raw"]
        assert "+2.0%" in entries[0]["alpha"]

    def test_resolve_leaves_premature_entry_pending(self, tmp_path):
        """#1169：fetch_returns 仍为 None 时不执行反思，
        记录保持待结算状态。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        reflector = MagicMock()
        with patch.object(settlement, "fetch_returns", return_value=(None, None, None, None)):
            settlement.settle_pending("NVDA", log, reflector, {})
        assert len(log.get_pending_entries()) == 1  # still pending
        reflector.reflect_on_final_decision.assert_not_called()


# ---------------------------------------------------------------------------
# 向投资组合经理注入历史上下文
# ---------------------------------------------------------------------------

class TestPortfolioManagerInjection:

    # 在初始状态注入 past_context

    def test_past_context_in_initial_state(self):
        propagator = Propagator()
        state = propagator.create_initial_state("NVDA", "2026-01-10", past_context="some context")
        assert "past_context" in state
        assert state["past_context"] == "some context"

    def test_past_context_defaults_to_empty(self):
        propagator = Propagator()
        state = propagator.create_initial_state("NVDA", "2026-01-10")
        assert state["past_context"] == ""

    # 投资组合经理的 Prompt

    def test_pm_prompt_includes_past_context(self):
        captured = {}
        llm = _structured_pm_llm(captured)
        pm_node = create_portfolio_manager(llm)
        state = _make_pm_state(past_context="[2026-01-05 | NVDA | Buy | +5.0% | +2.0% | 5d]\nGreat call.")
        pm_node(state)
        assert "Lessons from prior decisions and outcomes" in captured["prompt"]
        assert "Great call." in captured["prompt"]

    def test_pm_no_past_context_no_section(self):
        """past_context 为空时，投资组合经理的 Prompt 完全省略经验章节。"""
        captured = {}
        llm = _structured_pm_llm(captured)
        pm_node = create_portfolio_manager(llm)
        state = _make_pm_state(past_context="")
        pm_node(state)
        assert "Lessons from prior decisions" not in captured["prompt"]

    def test_pm_returns_rendered_markdown_with_rating(self):
        """结构化 PortfolioDecision 渲染为 Markdown，
        让 Memory Log、信号处理器及 CLI 无需额外模型调用就能解析。"""
        captured = {}
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="Build position gradually over the next two weeks.",
            investment_thesis="AI capex cycle remains intact; institutional flows constructive.",
            price_target=215.0,
            time_horizon="3-6 months",
        )
        llm = _structured_pm_llm(captured, decision)
        pm_node = create_portfolio_manager(llm)
        result = pm_node(_make_pm_state())
        md = result["final_trade_decision"]
        assert "**Rating**: Overweight" in md
        assert "**Executive Summary**: Build position gradually" in md
        assert "**Investment Thesis**: AI capex cycle" in md
        assert "**Price Target**: 215.0" in md
        assert "**Time Horizon**: 3-6 months" in md

    def test_pm_falls_back_to_freetext_when_structured_unavailable(self):
        """提供商不支持 with_structured_output 时，
        Agent 应回退到普通 invoke，并返回模型原始文本，不能阻塞流程。"""
        plain_response = "**Rating**: Sell\n\nExit ahead of guidance."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        pm_node = create_portfolio_manager(llm)
        result = pm_node(_make_pm_state())
        assert result["final_trade_decision"] == plain_response

    # get_past_context 的排序和数量限制

    def test_same_ticker_prioritised(self, tmp_path):
        """相同 ticker 的经验放在同标的章节，其他 ticker 放在跨标的章节。"""
        log = make_log(tmp_path)
        _resolve_entry(log, "NVDA", "2026-01-05", DECISION_BUY, "Momentum confirmed.")
        _resolve_entry(log, "AAPL", "2026-01-06", DECISION_SELL, "Overvalued.")
        result = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in result
        assert "Recent cross-ticker lessons" in result
        same_block, cross_block = result.split("Recent cross-ticker lessons")
        assert "NVDA" in same_block
        assert "AAPL" in cross_block

    def test_cross_ticker_reflection_only(self, tmp_path):
        """跨 ticker 经验只展示 REFLECTION，不展示完整 DECISION。"""
        log = make_log(tmp_path)
        _resolve_entry(log, "AAPL", "2026-01-06", DECISION_SELL, "Overvalued correction.")
        result = log.get_past_context("NVDA")
        assert "Overvalued correction." in result
        assert "Exit position immediately." not in result

    def test_n_same_limit_respected(self, tmp_path):
        """超过五条的同标的已结算记录只注入最近五条。"""
        log = make_log(tmp_path)
        for i in range(7):
            _resolve_entry(log, "NVDA", f"2026-01-{i+1:02d}", DECISION_BUY, f"Lesson {i}.")
        result = log.get_past_context("NVDA", n_same=5)
        lessons_present = sum(1 for i in range(7) if f"Lesson {i}." in result)
        assert lessons_present == 5

    def test_n_cross_limit_respected(self, tmp_path):
        """超过三条的跨标的已结算记录只注入最近三条。"""
        log = make_log(tmp_path)
        tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]
        for i, ticker in enumerate(tickers):
            _resolve_entry(log, ticker, f"2026-01-{i+1:02d}", DECISION_BUY, f"{ticker} lesson.")
        result = log.get_past_context("NVDA", n_cross=3)
        cross_count = sum(result.count(f"{t} lesson.") for t in tickers)
        assert cross_count == 3

    # 完整 A→B→C 集成链路

    def test_full_cycle_store_resolve_inject(self, tmp_path):
        """完整流程：保存待结算记录 → 根据结果结算 → PM 得到非空历史经验。"""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        assert len(log.get_pending_entries()) == 1
        assert log.get_past_context("NVDA") == ""
        log.update_with_outcome("NVDA", "2026-01-05", 0.05, 0.02, 5, "Correct call.")
        assert log.get_pending_entries() == []
        past_ctx = log.get_past_context("NVDA")
        assert past_ctx != ""
        assert "NVDA" in past_ctx
        assert "Correct call." in past_ctx
        assert "DECISION:" in past_ctx
        assert "REFLECTION:" in past_ctx


# ---------------------------------------------------------------------------
# 完全移除旧版 BM25 / FinancialSituationMemory
# ---------------------------------------------------------------------------

class TestLegacyRemoval:

    def test_financial_situation_memory_removed(self):
        """memory 模块不应再导出 FinancialSituationMemory。"""
        import tradingagents.memory as m
        assert not hasattr(m, "FinancialSituationMemory")

    def test_bm25_not_imported(self):
        """memory 模块命名空间中不应再包含 rank_bm25。"""
        import tradingagents.memory as m
        assert not hasattr(m, "BM25Okapi")

    def test_reflect_and_remember_removed(self):
        """TradingAgentsGraph 不应继续暴露 reflect_and_remember。"""
        assert not hasattr(TradingAgentsGraph, "reflect_and_remember")

    def test_portfolio_manager_no_memory_param(self):
        """create_portfolio_manager 只接收 llm；传入 memory= 应抛出 TypeError。"""
        mock_llm = MagicMock()
        create_portfolio_manager(mock_llm)
        with pytest.raises(TypeError):
            create_portfolio_manager(mock_llm, memory=MagicMock())

    def test_full_pipeline_no_regression(self, tmp_path):
        """重设计后 propagate() 完成运行并正确保存决策。"""
        import functools

        fake_state = {
            "final_trade_decision": "Rating: Buy\nBuy NVDA.",
            "final_rating": "Buy",
            "company_of_interest": "NVDA",
            "trade_date": "2026-01-10",
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {
                "bull_history": "", "bear_history": "", "history": "",
                "current_response": "",
            },
            "investment_plan": "",
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "", "conservative_history": "",
                "neutral_history": "", "history": "",
                "current_aggressive_response": "", "current_conservative_response": "",
                "current_neutral_response": "", "count": 1, "latest_speaker": "",
            },
        }
        mock_graph = MagicMock()
        mock_graph.memory_log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
        mock_graph.debug = False
        mock_graph.config = {"results_dir": str(tmp_path)}
        mock_graph.graph.invoke.return_value = fake_state
        mock_graph.propagator.create_initial_state.return_value = fake_state
        mock_graph.propagator.get_graph_args.return_value = {}
        # 绑定真实 _run_graph，使 propagate 调用该方法时执行实际写入，
        # 而不是使用 MagicMock 自动生成的替代方法。
        mock_graph._run_graph = functools.partial(
            TradingAgentsGraph._run_graph, mock_graph
        )
        mock_graph.record_decision = functools.partial(
            TradingAgentsGraph.record_decision, mock_graph
        )
        TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")
        entries = mock_graph.memory_log.load_entries()
        assert len(entries) == 1
        assert entries[0]["ticker"] == "NVDA"
        assert entries[0]["pending"] is True


@pytest.mark.unit
def test_a_failed_reflection_leaves_the_entry_pending_and_lets_the_run_start(tmp_path, monkeypatch):
    """新运行开始时会结算历史决策，反思需要调用 LLM；
    如果该调用短暂失败，不应终止本次新分析。"""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.memory import TradingMemoryLog

    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_log_path": str(tmp_path / "m.md")}
    graph.memory_log = TradingMemoryLog(graph.config)
    graph.memory_log.store_decision("NVDA", "2026-01-05", "Rating: Buy\n\nx")
    graph.memory_log.store_decision("NVDA", "2026-01-12", "Rating: Sell\n\ny")
    monkeypatch.setattr(settlement, "resolve_benchmark", lambda t, c: "SPY")
    monkeypatch.setattr(settlement, "fetch_returns",
                        lambda t, d, holding_days=5, benchmark=None: (0.01, 0.005, holding_days, "2026-01-19"))

    class _Reflector:
        calls = 0

        def reflect_on_final_decision(self, **kw):
            _Reflector.calls += 1
            if _Reflector.calls == 1:
                raise RuntimeError("provider timed out")
            return "second one worked"

    graph.reflector = _Reflector()

    graph.settle_pending("NVDA")  # must not raise

    entries = graph.memory_log.load_entries()
    assert [e["pending"] for e in entries] == [True, False]  # the failed one waits for next time


@pytest.mark.unit
def test_the_holding_window_is_configurable(tmp_path, monkeypatch):
    """设计为数月的决策不应在未经用户指定时用一周结果评分。"""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.memory import TradingMemoryLog

    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_log_path": str(tmp_path / "m.md"), "holding_period_days": 21}
    graph.memory_log = TradingMemoryLog(graph.config)
    graph.memory_log.store_decision("NVDA", "2026-01-05", "**Rating**: Buy\n\nx")
    monkeypatch.setattr(settlement, "resolve_benchmark", lambda t, c: "SPY")
    asked = {}

    def _returns(ticker, date, holding_days=5, benchmark=None):
        asked["holding_days"] = holding_days
        return 0.05, 0.02, holding_days, "2026-02-02"

    monkeypatch.setattr(settlement, "fetch_returns", _returns)
    graph.reflector = type("R", (), {"reflect_on_final_decision": lambda self, **kw: "lesson"})()

    graph.settle_pending("NVDA")

    assert asked["holding_days"] == 21
    assert graph.memory_log.load_entries()[0]["holding"] == "21d"


@pytest.mark.unit
def test_the_reflection_states_the_window_it_judges():
    """没有说明观察窗口就用一周 Alpha 判断长期论点，
    会把期限不匹配误写成决策失败的经验。"""
    from tradingagents.memory.reflection import Reflector

    prompt = Reflector(None)._system_prompt(holding_days=5)
    assert "5" in prompt and "trading day" in prompt


@pytest.mark.unit
def test_a_longer_window_asks_for_enough_price_history(monkeypatch):
    """交易日不是自然日：21 个交易日大约需要一个月行情，
    若只请求 28 个自然日可能使全部结果无法结算。"""
    asked = {}

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            asked["start"], asked["end"] = start, end
            import pandas as pd
            days = pd.bdate_range(start, end)
            return pd.DataFrame({"Close": range(len(days))}, index=days)

    monkeypatch.setattr("tradingagents.dataflows.vendors.yahoo.market.yf.Ticker", _Ticker)

    raw, alpha, days, resolved = settlement.fetch_returns("NVDA", "2026-06-01", 21, benchmark="SPY")

    assert days == 21 and resolved is not None, (raw, alpha, days, resolved)


@pytest.mark.unit
def test_concurrent_writers_keep_every_decision_and_outcome(tmp_path):
    """多个 Graph 同时使用同一日志写入和结算时，不得丢失任何写入。"""
    import threading

    path = tmp_path / "memory.md"
    tickers = [f"T{i}" for i in range(8)]
    dates = [f"2026-01-{day:02d}" for day in range(1, 13)]
    errors = []

    def work(ticker):
        log = TradingMemoryLog({"memory_log_path": str(path)})
        try:
            for date in dates:
                log.store_decision(ticker, date, "**Rating**: Buy")
                log.update_with_outcome(ticker, date, 0.01, 0.005, 5, "held up")
        except Exception as exc:  # noqa: BLE001 — collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in tickers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    entries = TradingMemoryLog({"memory_log_path": str(path)}).load_entries()
    assert errors == []
    assert len(entries) == len(tickers) * len(dates)
    assert all(not e["pending"] for e in entries)


def _closes(by_symbol):
    """模拟 get_closes，按各 ticker 和请求日期范围返回价格序列。"""
    def get_closes(symbol, start_date, end_date):
        series = by_symbol[symbol]
        return series[(series.index >= start_date) & (series.index < end_date)]
    return get_closes


@pytest.mark.unit
class TestSettlementWindow:
    """不论各自交易日历如何，股票和基准都需要使用一致的比较日期。"""

    def test_a_seven_day_market_is_scored_against_the_benchmark_over_the_same_dates(self, monkeypatch):
        coin = pd.Series([100.0 + i for i in range(10)], index=pd.date_range("2026-01-03", periods=10, freq="D"))
        weekdays = pd.bdate_range("2025-12-29", "2026-01-16")
        spy = pd.Series([400.0 + 2 * i for i in range(len(weekdays))], index=weekdays)
        monkeypatch.setattr(settlement, "get_closes", _closes({"BTC-USD": coin, "SPY": spy}))

        raw, alpha, days, resolved = settlement.fetch_returns("BTC-USD", "2026-01-03", 5, "SPY")

        # 周六 3 日决策，加密资产从周日 4 日持有到周五 9 日；
        # SPY 同期价格区间为周五 2 日至周五 9 日。
        assert resolved == "2026-01-09"
        assert raw == pytest.approx(106.0 / 101.0 - 1)
        assert alpha == pytest.approx(raw - (spy["2026-01-09"] / spy["2026-01-02"] - 1))

    def test_bars_stamped_in_different_time_zones_meet_on_their_dates(self, monkeypatch):
        # Yahoo 对加密资产使用 UTC 午夜，对 SPY 使用纽约午夜，
        # 两者可能相差五小时，但仍对应同一个自然日期，不能错算为次日。
        coin = pd.Series([100.0 + i for i in range(10)],
                         index=pd.date_range("2026-01-03", periods=10, freq="D", tz="UTC"))
        weekdays = pd.bdate_range("2025-12-29", "2026-01-16", tz="America/New_York")
        spy = pd.Series([400.0 + 2 * i for i in range(len(weekdays))], index=weekdays)
        monkeypatch.setattr(settlement, "get_closes", _closes({"BTC-USD": coin, "SPY": spy}))

        raw, alpha, _, resolved = settlement.fetch_returns("BTC-USD", "2026-01-03", 5, "SPY")

        assert resolved == "2026-01-09"
        spy_by_day = spy.tz_localize(None)
        assert alpha == pytest.approx(raw - (spy_by_day["2026-01-09"] / spy_by_day["2026-01-02"] - 1))

    def test_the_outcome_waits_for_the_benchmark_to_trade_through_the_window(self, monkeypatch):
        coin = pd.Series([100.0 + i for i in range(10)], index=pd.date_range("2026-01-03", periods=10, freq="D"))
        spy = pd.Series([400.0, 401.0, 402.0], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))
        monkeypatch.setattr(settlement, "get_closes", _closes({"BTC-USD": coin, "SPY": spy}))

        assert settlement.fetch_returns("BTC-USD", "2026-01-03", 5, "SPY") == (None, None, None, None)

    def test_a_missing_close_is_skipped_not_scored(self, monkeypatch):
        idx = pd.bdate_range("2026-01-05", periods=8)
        stock = pd.Series([100.0, float("nan"), 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], index=idx)
        spy = pd.Series([400.0 + i for i in range(8)], index=idx)
        monkeypatch.setattr(settlement, "get_closes", _closes({"NVDA": stock, "SPY": spy}))

        raw, alpha, _, resolved = settlement.fetch_returns("NVDA", "2026-01-05", 5, "SPY")

        # 从决策后的首个真实收盘价（周三 7 日）入场，跳过缺失的周二 6 日。
        assert raw == pytest.approx(106.0 / 101.0 - 1)
        assert resolved == "2026-01-14"
        assert alpha == pytest.approx(raw - (spy["2026-01-14"] / spy["2026-01-07"] - 1))


@pytest.mark.unit
def test_a_close_that_is_not_a_price_is_skipped_not_scored(monkeypatch):
    """收盘价为零会导致无限收益率，不能将其写入 Memory Log。"""
    idx = pd.bdate_range("2026-01-05", periods=8)
    stock = pd.Series([0.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0], index=idx)
    spy = pd.Series([400.0 + i for i in range(8)], index=idx)
    monkeypatch.setattr(settlement, "get_closes", _closes({"NVDA": stock, "SPY": spy}))

    raw, alpha, _, _ = settlement.fetch_returns("NVDA", "2026-01-02", 5, "SPY")

    assert raw == pytest.approx(105.0 / 100.0 - 1)   # the window opens on the first real close


@pytest.mark.unit
class TestSettleAll:
    """调度器停止分析某个 ticker 后仍应能结算其历史决策（#1445）。"""

    def _setup(self, tmp_path, monkeypatch):
        log = make_log(tmp_path)
        log.store_decision("AAPL", "2026-01-05", DECISION_BUY)
        log.store_decision("MSFT", "2026-01-05", DECISION_BUY)
        idx = pd.bdate_range("2026-01-02", periods=12)
        series = pd.Series([100.0 + i for i in range(12)], index=idx)
        monkeypatch.setattr(settlement, "get_closes", _closes({"AAPL": series, "MSFT": series, "SPY": series}))

        class _Reflector:
            def reflect_on_final_decision(self, **kwargs):
                return "Held up."
        return log, _Reflector()

    def test_every_ticker_with_a_due_decision_is_settled(self, tmp_path, monkeypatch):
        log, reflector = self._setup(tmp_path, monkeypatch)

        result = settlement.settle_all_pending(log, reflector, {"holding_period_days": 5})

        assert sorted(result.settled) == [("AAPL", "2026-01-05"), ("MSFT", "2026-01-05")]
        assert result.failed == []
        assert not log.get_pending_entries()

    def test_a_failed_reflection_is_reported_and_left_pending(self, tmp_path, monkeypatch):
        log, _ = self._setup(tmp_path, monkeypatch)

        class _Down:
            def reflect_on_final_decision(self, **kwargs):
                raise RuntimeError("provider down")

        result = settlement.settle_pending("AAPL", log, _Down(), {"holding_period_days": 5})

        assert result.settled == [] and [f[:2] for f in result.failed] == [("AAPL", "2026-01-05")]
        assert {e["ticker"] for e in log.get_pending_entries()} == {"AAPL", "MSFT"}


@pytest.mark.unit
def test_the_graph_settles_every_ticker_under_its_own_config(tmp_path, monkeypatch):
    from tradingagents.dataflows.config import get_config

    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"holding_period_days": 5, "output_language": "Deutsch"}
    graph.memory_log, graph.reflector = make_log(tmp_path), object()
    seen = []
    monkeypatch.setattr(settlement, "settle_all_pending",
                        lambda log, reflector, config, wait=True: seen.append(get_config()["output_language"])
                        or settlement.Settlement(settled=[("AAPL", "2026-01-05")]))

    assert graph.settle_all_pending().settled == [("AAPL", "2026-01-05")]
    assert seen == ["Deutsch"]


@pytest.mark.unit
class TestSettlementOnceAndDurably:
    """结算反思会产生 LLM 费用，因此每条决策只能反思一次；
    成功部分应保留，无法完成的部分需报告（#1428、#1445）。"""

    def _log(self, tmp_path, monkeypatch, *dates):
        log = make_log(tmp_path)
        for date in dates:
            log.store_decision("AAPL", date, DECISION_BUY)
        series = pd.Series([100.0 + i for i in range(40)], index=pd.bdate_range("2026-01-02", periods=40))
        monkeypatch.setattr(settlement, "get_closes", _closes({"AAPL": series, "SPY": series}))
        return log

    def test_two_runs_settling_at_once_reflect_once(self, tmp_path, monkeypatch):
        import threading
        import time

        self._log(tmp_path, monkeypatch, "2026-01-05")
        calls = []

        class _Slow:
            def reflect_on_final_decision(self, **kwargs):
                calls.append(1)
                time.sleep(0.2)
                return "Held up."

        results = []
        threads = [threading.Thread(target=lambda: results.append(
            settlement.settle_pending("AAPL", make_log(tmp_path), _Slow(), {"holding_period_days": 5})))
            for _ in range(2)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        assert len(calls) == 1
        assert sorted(len(r.settled) for r in results) == [0, 1]

    def test_an_interrupted_pass_keeps_the_outcomes_it_finished(self, tmp_path, monkeypatch):
        log = self._log(tmp_path, monkeypatch, "2026-01-05", "2026-01-12")

        class _Interrupted:
            calls = 0

            def reflect_on_final_decision(self, **kwargs):
                _Interrupted.calls += 1
                if _Interrupted.calls == 2:
                    raise KeyboardInterrupt
                return "Held up."

        with pytest.raises(KeyboardInterrupt):
            settlement.settle_pending("AAPL", log, _Interrupted(), {"holding_period_days": 5})

        assert [e["date"] for e in log.get_pending_entries()] == ["2026-01-12"]

    def test_a_price_request_that_fails_is_reported_not_taken_for_not_due(self, tmp_path, monkeypatch):
        from tradingagents.dataflows.errors import VendorUnavailableError

        log = self._log(tmp_path, monkeypatch, "2026-01-05")

        def down(*args, **kwargs):
            raise VendorUnavailableError("Yahoo Finance request failed: TimeoutError")

        monkeypatch.setattr(settlement, "get_closes", down)

        result = settlement.settle_pending("AAPL", log, object(), {"holding_period_days": 5})

        assert result.settled == [] and "prices unavailable" in result.failed[0][2]


@pytest.mark.unit
def test_any_failure_to_price_a_decision_is_reported(tmp_path, monkeypatch):
    log = make_log(tmp_path)
    log.store_decision("AAPL", "2026-01-05", DECISION_BUY)

    def broken(*args, **kwargs):
        raise RuntimeError("vendor outage")

    monkeypatch.setattr(settlement, "get_closes", broken)

    result = settlement.settle_pending("AAPL", log, object(), {"holding_period_days": 5})

    assert result.settled == [] and "vendor outage" in result.failed[0][2]


@pytest.mark.unit
def test_a_run_does_not_wait_on_another_settlement_pass(tmp_path):
    """其他结算流程正在处理同一日志时，本次步骤不必等待，继续运行。"""
    import threading
    import time

    log = make_log(tmp_path)
    log.store_decision("AAPL", "2026-01-05", DECISION_BUY)
    held, release = threading.Event(), threading.Event()

    def other_pass():
        with make_log(tmp_path).settling():
            held.set()
            release.wait(5)

    worker = threading.Thread(target=other_pass)
    worker.start()
    held.wait(5)
    started = time.monotonic()
    result = settlement.settle_all_pending(log, object(), {"holding_period_days": 5}, wait=False)
    elapsed = time.monotonic() - started
    release.set()
    worker.join()

    assert elapsed < 1 and result.settled == [] and result.failed == []
