"""历史基本面分析不能泄漏当前公司概况（#1300）。

Yahoo 的 Ticker.info 和 Alpha Vantage 的 OVERVIEW 仅提供当前值：
市值、估值、近一年价格区间及 TTM 指标随行情变化，
公司名、板块和行业也可能发生更名或调整。
由于缺少历史版本，不能把今天的这些字段注入过去的运行。
两个提供商共用 date_window.withhold_live_profile，
防止切换数据源绕过限制；当日运行不受影响。所有请求均使用 mock。"""
from __future__ import annotations

from unittest import mock

import pytest

from tradingagents.dataflows import date_window
from tradingagents.dataflows.vendors.alpha_vantage import fundamentals as av
from tradingagents.dataflows.vendors.yahoo import (
    fundamentals as yahoo_fundamentals,
    market as yahoo_market,
)

_TODAY = "2026-09-07"
_PAST = "2024-05-10"

# 公司概况混合了看似稳定的身份字段和随行情变化的字段。
_INFO = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_500_000_000_000,
    "trailingPE": 34.2,
    "fiftyTwoWeekHigh": 260.1,
    "totalRevenue": 391_000_000_000,
}

# 这些值不能传给历史运行。
_LEAKY = ("3500000000000", "34.2", "260.1", "391000000000",
          "Apple Inc.", "Technology", "Consumer Electronics")


def _yf(as_of_date, info=_INFO, today=_TODAY):
    with mock.patch.object(date_window, "get_current_date", return_value=today), \
         mock.patch.object(yahoo_fundamentals, "yf_retry", lambda fn: info), \
         mock.patch.object(yahoo_market.yf, "Ticker"):
        return yahoo_fundamentals.get_fundamentals("AAPL", as_of_date)


def _av(as_of_date, today=_TODAY):
    """通过 mock Alpha Vantage 请求验证历史泄漏检查。"""
    with mock.patch.object(date_window, "get_current_date", return_value=today), \
         mock.patch.object(av, "_make_api_request",
                           return_value="MarketCapitalization: 3500000000000") as req:
        return av.get_fundamentals("AAPL", as_of_date), req


@pytest.mark.unit
class TestYFinanceHistoricalRun:
    def test_no_profile_value_survives(self):
        out = _yf(_PAST)
        for leaked in _LEAKY:
            assert leaked not in out, f"leaked live-profile value {leaked!r}"

    def test_states_the_as_of_date_and_explains_itself(self):
        # 必须告诉分析师缺失数值的原因，
        # 避免模型将数据缺口误判为真实市场信号或自行编造。
        out = _yf(_PAST)
        assert f"Point-in-time as of: {_PAST}" in out
        assert "withheld" in out
        assert _PAST in out and _TODAY not in out

    def test_no_wall_clock_retrieval_stamp(self):
        # 旧报告标题使用 datetime.now()，暴露了时间泄漏问题。
        assert "Data retrieved on:" not in _yf(_PAST)

    def test_the_request_is_not_even_made(self):
        # 历史调用的响应最终会被丢弃，因此提前跳过还可节省 API 请求额度。
        with mock.patch.object(date_window, "get_current_date", return_value=_TODAY), \
             mock.patch.object(yahoo_market.yf, "Ticker") as tk:
            yahoo_fundamentals.get_fundamentals("AAPL", _PAST)
        tk.assert_not_called()


@pytest.mark.unit
class TestAlphaVantageHistoricalRun:
    """另一个基本面提供商也必须遵守该规则，
    否则切换 data_vendors["fundamental_data"] 就会重新引入泄漏。"""

    def test_overview_is_withheld(self):
        out, _ = _av(_PAST)
        assert "3500000000000" not in out
        assert "withheld" in out
        assert f"Point-in-time as of: {_PAST}" in out

    def test_the_api_call_is_not_made(self):
        _, req = _av(_PAST)
        req.assert_not_called()

    def test_live_run_still_calls_the_api(self):
        out, req = _av(_TODAY)
        req.assert_called_once()
        assert "3500000000000" in out


@pytest.mark.unit
class TestLiveRunUnchanged:
    def test_yfinance_current_date_returns_the_full_profile(self):
        out = _yf(_TODAY)
        for value in _LEAKY:
            assert value in out
        assert "withheld" not in out

    def test_yfinance_absent_curr_date_returns_the_full_profile(self):
        out = _yf(None)
        assert "Market Cap: 3500000000000" in out
        assert "withheld" not in out


@pytest.mark.unit
class TestNoUsableFieldsStillRaises:
    def test_stub_payload_raises_no_market_data(self, monkeypatch):
        # yfinance 对未知标的可能返回 {"trailingPegRatio": None}；
        # 实时运行应将它视为无数据，不能输出只有标题的空报告。
        from tradingagents.dataflows.errors import NoMarketDataError
        from tradingagents.dataflows.vendors.yahoo import common

        monkeypatch.setattr(common, "vendor_reachable", lambda url: True)

        with pytest.raises(NoMarketDataError):
            _yf(_TODAY, info={"trailingPegRatio": None})
