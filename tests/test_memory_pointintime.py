"""回测中的 Memory Log 经验必须满足时点一致性（#1251）。

此前 get_past_context 不区分分析日，可能让历史运行使用未来才产生的收益反思。
已结算记录现在保存结果变得可知的日期（``resolved:``），
调用 get_past_context(as_of=...) 时按照该日期过滤。
没有结算日期的旧记录在历史查询中保守排除，避免信息泄漏。"""
from __future__ import annotations

import pytest

from tradingagents.memory import TradingMemoryLog


def _log(tmp_path):
    return TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})


def _resolve(log, ticker, date, resolution_date, reflection):
    log.store_decision(ticker, date, f"Rating: Buy\n{reflection}")
    log.update_with_outcome(
        ticker, date, 0.05, 0.02, 5, reflection, resolution_date=resolution_date,
    )


@pytest.mark.unit
def test_resolution_date_is_stored_and_parsed(tmp_path):
    log = _log(tmp_path)
    _resolve(log, "NVDA", "2026-01-05", "2026-01-10", "outcome known 01-10")
    entry = log.load_entries()[0]
    assert entry["resolved"] == "2026-01-10"
    assert "resolved:2026-01-10" in (tmp_path / "mem.md").read_text()


@pytest.mark.unit
def test_as_of_excludes_lessons_resolved_after_the_run_date(tmp_path):
    log = _log(tmp_path)
    # 01-05 作出决策，01-10 才知道结果。
    _resolve(log, "NVDA", "2026-01-05", "2026-01-10", "great trade")

    # 截至 01-07 的运行不能读取这条未来结果。
    assert log.get_past_context("NVDA", as_of="2026-01-07") == ""
    # 01-10 或之后的运行可以读取。
    assert "great trade" in log.get_past_context("NVDA", as_of="2026-01-10")
    assert "great trade" in log.get_past_context("NVDA", as_of="2026-02-01")


@pytest.mark.unit
def test_no_as_of_is_unfiltered_live_behavior(tmp_path):
    log = _log(tmp_path)
    _resolve(log, "NVDA", "2026-01-05", "2026-01-10", "great trade")
    # 实时运行未提供 as_of 时继续展示该经验。
    assert "great trade" in log.get_past_context("NVDA")


@pytest.mark.unit
def test_legacy_entry_without_resolution_date_excluded_in_backtest(tmp_path):
    log = _log(tmp_path)
    # 模拟迁移前已结算但缺少 resolution_date 的旧记录。
    log.store_decision("NVDA", "2026-01-05", "Rating: Buy\nlegacy lesson")
    log.update_with_outcome("NVDA", "2026-01-05", 0.05, 0.02, 5, "legacy lesson")
    entry = log.load_entries()[0]
    assert entry["resolved"] is None

    # 保守处理：无法证明当时已经知道结果，因此历史查询中排除。
    assert log.get_past_context("NVDA", as_of="2026-06-01") == ""
    # 但实时、未过滤的运行仍可读取。
    assert "legacy lesson" in log.get_past_context("NVDA")


@pytest.mark.unit
def test_cross_ticker_lessons_are_also_gated(tmp_path):
    log = _log(tmp_path)
    _resolve(log, "AAPL", "2026-01-05", "2026-01-10", "cross lesson")
    # 在结算前查询其他 ticker，也不能泄漏跨标的历史经验。
    assert log.get_past_context("NVDA", as_of="2026-01-07") == ""
    assert "cross lesson" in log.get_past_context("NVDA", as_of="2026-01-10")


@pytest.mark.unit
def test_memory_as_of_gates_historical_but_not_live():
    # Graph 仅在历史交易日期执行过滤；今天的运行传入 None，
    # 保留实时逻辑及对旧记录的兼容（#1251）。
    from datetime import datetime, timedelta

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    g = object.__new__(TradingAgentsGraph)
    past = "2024-01-01"
    today = datetime.now().strftime("%Y-%m-%d")
    future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    assert g._memory_as_of(past) == past       # backtest -> filter on the trade date
    assert g._memory_as_of(today) is None      # live -> no filter
    assert g._memory_as_of(future) is None     # future-dated run -> no filter
