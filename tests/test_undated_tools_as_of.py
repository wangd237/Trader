"""内部人交易和预测市场赔率必须遵守本次运行的交易日期。

工具本身不从模型读取日期，而是从 Graph 状态注入 trade_date。
内部人交易按披露时间过滤；Polymarket 只提供实时赔率，历史运行应屏蔽。"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from tradingagents.agents import tools
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.vendors import polymarket
from tradingagents.dataflows.vendors.yahoo import (
    fundamentals as yahoo_fundamentals,
    market as yahoo_market,
)


def _insider_frame(*dates):
    return pd.DataFrame({
        "Shares": [100] * len(dates),
        "Text": [f"Sale at price {100 + i} per share." for i in range(len(dates))],
        "Start Date": pd.to_datetime(list(dates)),
    })


def _yf_insider(frame, curr_date):
    ticker = mock.Mock(insider_transactions=frame)
    with mock.patch.object(yahoo_market.yf, "Ticker", return_value=ticker):
        return yahoo_fundamentals.get_insider_transactions("AAPL", curr_date)


@pytest.mark.unit
def test_yfinance_insider_without_a_date_is_unfiltered():
    out = _yf_insider(_insider_frame("2026-09-08", "2025-01-10"), None)
    assert "2026-09-08" in out and "2025-01-10" in out


@pytest.mark.unit
def test_polymarket_withholds_live_odds_from_a_historical_run():
    with mock.patch.object(polymarket, "_request", side_effect=AssertionError("must not fetch")):
        out = polymarket.get_prediction_markets("Fed rate cut", as_of_date="2025-06-01")
    assert "withheld" in out


@pytest.mark.unit
def test_polymarket_serves_a_current_run():
    with mock.patch.object(polymarket, "_request", return_value={"events": []}) as req:
        polymarket.get_prediction_markets("Fed rate cut", as_of_date=get_current_date())
    req.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("tool", [tools.get_insider_transactions,
                                  tools.get_prediction_markets], ids=lambda t: t.name)
def test_trade_date_is_injected_not_model_visible(tool):
    assert "trade_date" in tool.func.__code__.co_varnames
    props = tool.tool_call_schema.model_json_schema()["properties"]
    assert "trade_date" not in props and "curr_date" not in props


# --- the instrument's identity -------------------------------------------------

@pytest.mark.unit
def test_a_historical_run_gets_the_current_name_only_as_an_identifier(monkeypatch):
    """公司概况来自今天，但名称仍可用于区分公司（#814），
    历史运行只保留并注明该名称是当前值，
    不能传入当时未必成立的板块、行业和交易所信息。"""
    from tradingagents.agents.context import build_instrument_context

    identity = {"company_name": "Example Corp", "sector": "Technology",
                "industry": "Software", "exchange": "NMS"}

    historical = build_instrument_context("EXMP", "stock", identity, trade_date="2024-03-14")
    assert "Example Corp" in historical and "current name" in historical
    assert "2024-03-14" in historical
    for later in ("Technology", "Software", "NMS"):
        assert later not in historical


@pytest.mark.unit
def test_a_current_run_is_not_cluttered_with_a_vintage_note(monkeypatch):
    from tradingagents.agents.context import build_instrument_context
    from tradingagents.dataflows.date_window import get_current_date

    today = build_instrument_context("EXMP", "stock", {"company_name": "Example Corp"},
                                     trade_date=get_current_date())
    assert "Example Corp" in today
    assert "current name" not in today


@pytest.mark.unit
def test_an_indicator_that_could_not_be_read_is_not_shown_as_a_blank_value():
    """按日读取失败时如果直接返回空字符串，
    指标表会显示“日期: ”，误导模型认为当天真实没有指标，
    而实际只是请求失败。"""
    from tradingagents.dataflows.errors import VendorError

    with mock.patch.object(yahoo_market, "get_stock_stats",
                           side_effect=RuntimeError("cache parse failed")), \
            pytest.raises(VendorError):
        yahoo_market.get_stockstats_indicator("AAPL", "rsi", "2026-05-08")


@pytest.mark.unit
@pytest.mark.parametrize("func, args", [
    # 历史日期在请求之前就屏蔽实时概况，
    # 因此基本面测试只在符合条件的日期发起请求。
    ("get_fundamentals", ("AAPL", None)),
    # 仅当日运行可请求无披露时间的报表；历史运行应屏蔽它们。
    ("get_balance_sheet", ("AAPL", "annual", get_current_date())),
    ("get_cashflow", ("AAPL", "annual", get_current_date())),
    ("get_income_statement", ("AAPL", "annual", get_current_date())),
    ("get_insider_transactions", ("AAPL", get_current_date())),
])
def test_a_yfinance_failure_is_a_vendor_error_not_a_report(func, args):
    """如果把故障直接返回为普通文本，路由器会将其视为成功响应并停止回退；
    这是默认 yfinance 路径尤其需要避免的情况。"""
    from tradingagents.dataflows.errors import VendorError

    with mock.patch.object(yahoo_market.yf, "Ticker", side_effect=RuntimeError("yahoo hiccup")), \
            pytest.raises(VendorError):
        getattr(yahoo_fundamentals, func)(*args)


@pytest.mark.unit
@pytest.mark.parametrize("func, args", [
    ("get_news_yfinance", ("AAPL", "2026-08-25", "2026-09-01")),
    ("get_global_news_yfinance", ("2026-09-01", 7, 5)),
])
def test_a_yfinance_news_failure_is_a_vendor_error_not_a_report(func, args):
    from tradingagents.dataflows.errors import VendorError
    from tradingagents.dataflows.vendors.yahoo import news as yahoo_news

    target = "Ticker" if "global" not in func else "Search"
    with mock.patch.object(yahoo_news.yf, target, side_effect=RuntimeError("yahoo hiccup")), \
            pytest.raises(VendorError):
        getattr(yahoo_news, func)(*args)


@pytest.mark.unit
def test_an_unreachable_vendor_is_not_reported_as_a_missing_symbol(monkeypatch):
    """yfinance 无法连接 Yahoo 时可能只返回空表而不抛异常，
    不能据此宣称 AAPL 没有财报，实际是服务不可达。"""
    import pandas as pd

    from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
    from tradingagents.dataflows.vendors.yahoo import common

    empty = mock.Mock(quarterly_balance_sheet=pd.DataFrame(), balance_sheet=pd.DataFrame())
    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: empty)

    monkeypatch.setattr(common, "vendor_reachable", lambda url: False)
    with pytest.raises(VendorUnavailableError, match="unreachable"):
        yahoo_fundamentals.get_balance_sheet("AAPL", "annual", get_current_date())

    monkeypatch.setattr(common, "vendor_reachable", lambda url: True)
    with pytest.raises(NoMarketDataError):
        yahoo_fundamentals.get_balance_sheet("AAPL", "annual", get_current_date())


@pytest.mark.unit
def test_every_vendor_unavailable_says_so_rather_than_crashing(monkeypatch):
    """限流或整条回退链不可达时，旧实现会抛 RuntimeError 并终止运行，
    但没有明确解释是提供商故障。"""
    from tradingagents.dataflows import router
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.errors import VendorUnavailableError

    def _down(*a, **k):
        raise VendorUnavailableError("Yahoo Finance is unreachable")

    set_config({"data_vendors": {"fundamental_data": "yfinance"}})
    monkeypatch.setitem(router.VENDOR_METHODS["get_balance_sheet"], "yfinance", _down)

    out = router.route_to_vendor("get_balance_sheet", "AAPL", "annual", "2026-09-01")

    assert "unavailable" in out.lower() and "unreachable" in out.lower()
    assert "delisted" not in out.lower()  # not a claim about the symbol


@pytest.mark.unit
def test_the_price_path_also_tells_an_outage_from_an_unknown_symbol(monkeypatch):
    """价格是最常调用的数据，故障不能被误写成标的退市。"""
    import pandas as pd

    from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
    from tradingagents.dataflows.vendors.yahoo import common

    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: mock.Mock(history=lambda **k: pd.DataFrame()))

    monkeypatch.setattr(common, "vendor_reachable", lambda url: False)
    with pytest.raises(VendorUnavailableError, match="unreachable"):
        yahoo_market.get_YFin_data_online("AAPL", "2026-09-01", "2026-09-10")

    monkeypatch.setattr(common, "vendor_reachable", lambda url: True)
    with pytest.raises(NoMarketDataError):
        yahoo_market.get_YFin_data_online("AAPL", "2026-09-01", "2026-09-10")


@pytest.mark.unit
@pytest.mark.parametrize("func, args", [
    pytest.param(f, a, id=f.__name__) for f, a in (
        (yahoo_market.get_YFin_data_online, ("AAPL", "2025-06-02", "2025-06-06")),
        (yahoo_fundamentals.get_balance_sheet, ("AAPL", "quarterly", "2025-06-06")),
        (yahoo_fundamentals.get_cashflow, ("AAPL", "quarterly", "2025-06-06")),
        (yahoo_fundamentals.get_income_statement, ("AAPL", "quarterly", "2025-06-06")),
        (yahoo_fundamentals.get_insider_transactions, ("AAPL", "2025-06-06")),
    )
])
def test_a_historical_run_is_not_told_todays_date(func, args):
    """带当前墙上时间的标题会泄漏回测实际执行日期。"""
    from datetime import date

    statement = pd.DataFrame({pd.Timestamp("2025-03-31"): [1.0]}, index=["Total Assets"])
    prices = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
                          index=pd.DatetimeIndex(["2025-06-02"], name="Date"))
    ticker = mock.Mock(quarterly_balance_sheet=statement, quarterly_cashflow=statement,
                       quarterly_income_stmt=statement,
                       insider_transactions=_insider_frame("2025-05-30"),
                       history=lambda **k: prices)
    with mock.patch.object(yahoo_market.yf, "Ticker", return_value=ticker):
        out = func(*args)

    assert date.today().isoformat() not in out


def _dates_after(text: str, cutoff: str) -> list[str]:
    import re
    return [d for d in re.findall(r"\d{4}-\d{2}-\d{2}", text) if d > cutoff]


@pytest.mark.unit
def test_an_unavailable_notice_names_no_date_after_the_run():
    """数据缺失提示中曾显示数据源覆盖起点或今天日期；
    两者可能晚于历史运行日期，造成时间信息泄漏。"""
    from tradingagents.agents.context import build_instrument_context
    from tradingagents.dataflows.date_window import (
        coverage_gap,
        get_current_date,
        withhold_live_profile,
    )

    today = get_current_date()
    notices = [
        coverage_gap([pd.Timestamp(today, tz="UTC")], "2025-01-01", "2025-01-07", "Feed", "news"),
        withhold_live_profile("2025-01-07", "AAPL"),
        _yf_insider(_insider_frame(today), "2025-01-07"),
        build_instrument_context("EXMP", "stock", {"company_name": "Example"}, trade_date="2025-01-07"),
    ]
    for notice in notices:
        assert _dates_after(notice, "2025-01-07") == [], notice
