"""Offline tests for the first A-share vendor integration."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.vendors import ashare


def _bars(volume):
    return pd.DataFrame([{
        "date": "2026-09-30", "open": 11.50, "high": 11.66,
        "low": 11.43, "close": 11.57, "volume": volume,
        "amount": 1_206_000_000,
    }])


def test_sz000_volume_conversion(monkeypatch):
    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: _bars(1_045_357)),
    )
    report = ashare.get_stock_data("000001.SZ", "2026-09-01", "2026-09-30")
    assert "104535700" in report
    assert "# Source: Tencent" in report


def test_volume_not_doubled_if_already_shares(monkeypatch):
    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: _bars(104_535_700)),
    )
    report = ashare.get_stock_data("000001.SZ", "2026-09-01", "2026-09-30")
    assert "104535700" in report
    assert "10453570000" not in report


def test_disclosure_as_of_filters_future_statement(monkeypatch):
    df = pd.DataFrame([
        {"报告日": "20260930", "公告日期": "20261021", "营业收入": 300},
        {"报告日": "20260630", "公告日期": "20260815", "营业收入": 200},
    ])
    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_financial_report_sina=lambda **kwargs: df),
    )
    report = ashare.get_income_statement("300750.SZ", "quarterly", "2026-09-30")
    assert "20260630" in report
    assert "20260930" not in report
    assert "20260815" in report


def test_invalid_symbol_rejected():
    with pytest.raises(ValueError, match="Not an A-share"):
        ashare.get_stock_data("AAPL", "2026-09-01", "2026-09-30")


def test_router_registers_ashare_vendor():
    from tradingagents.dataflows.router import VENDOR_METHODS
    for method in ("get_stock_data", "get_balance_sheet", "get_cashflow", "get_income_statement"):
        assert "ashare" in VENDOR_METHODS[method]


def test_router_automatically_selects_ashare_without_user_configuration(monkeypatch):
    from tradingagents.dataflows import router

    called = []
    monkeypatch.setattr(ashare, "get_stock_data", lambda *args: called.append(args) or "A-SHARE")
    monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "ashare", ashare.get_stock_data)
    result = router.route_to_vendor("get_stock_data", "600519.SH", "2026-09-01", "2026-09-30")
    assert result == "A-SHARE"
    assert called[0][0] == "600519.SH"


def test_cninfo_announcement_cuts_future_and_handles_empty(monkeypatch):
    df = pd.DataFrame([
        {"代码": "600519", "公告标题": "old", "公告时间": "2026-09-29",
         "公告链接": "http://www.cninfo.com.cn/new/disclosure/detail?announcementId=123&announcementTime=2026-09-29"},
        {"代码": "600519", "公告标题": "future", "公告时间": "2026-10-21",
         "公告链接": "http://www.cninfo.com.cn/new/disclosure/detail?announcementId=124&announcementTime=2026-10-21"},
    ])
    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_zh_a_disclosure_report_cninfo=lambda **kwargs: df),
    )
    monkeypatch.setattr(ashare, "_pdf_text_from_announcement", lambda link: "Official PDF text " * 8)
    result = ashare.get_news("600519.SH", "2026-09-01", "2026-09-30")
    assert "old" in result
    assert "future" not in result

    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_zh_a_disclosure_report_cninfo=lambda **kwargs: pd.DataFrame()),
    )
    assert "NO_NEWS_AVAILABLE" in ashare.get_news("600519.SH", "2026-09-01", "2026-09-30")


def test_ashare_snapshot_uses_domestic_rows(monkeypatch):
    from tradingagents.agents import tools as agent_tools

    frame = pd.DataFrame([{
        "Date": pd.Timestamp("2026-09-30"), "Open": 11.5,
        "High": 11.7, "Low": 11.4, "Close": 11.57, "Volume": 104_535_700,
    }])
    monkeypatch.setattr(ashare, "load_ohlcv", lambda symbol, cutoff: frame)
    monkeypatch.setattr(
        agent_tools, "build_verified_market_snapshot",
        lambda symbol, cutoff, look_back_days, data=None: (
            "domestic" if data is frame else "wrong"
        ),
    )
    assert agent_tools.get_verified_market_snapshot.func(
        "000001.SZ", "2026-09-30"
    ) == "domestic"


def test_ashare_vendor_error_becomes_unavailable_sentinel(monkeypatch):
    from tradingagents.dataflows import router
    from tradingagents.dataflows.errors import VendorUnavailableError

    def offline(*args):
        raise VendorUnavailableError("Simulated outage")

    monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "ashare", offline)
    result = router.route_to_vendor(
        "get_stock_data", "600519.SH", "2026-09-01", "2026-09-30"
    )
    assert result.startswith("DATA_UNAVAILABLE:")


def test_identity_falls_back_to_official_cninfo_name(monkeypatch):
    def failed_profile(**kwargs):
        raise ConnectionError("Simulated Eastmoney proxy failure")

    def announcements(**kwargs):
        return pd.DataFrame([{"代码": "600519", "简称": "贵州茅台"}])

    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(
            stock_individual_info_em=failed_profile,
            stock_zh_a_disclosure_report_cninfo=announcements,
        ),
    )
    assert ashare.company_identity("600519.SH") == {"company_name": "贵州茅台"}


def test_dividend_evidence_uses_official_implementation(monkeypatch):
    df = pd.DataFrame([
        {"公告标题": "2026年半年度利润分配方案", "公告时间": pd.Timestamp("2026-08-15"),
         "公告链接": "proposal"},
        {"公告标题": "2026年半年度权益分派实施公告", "公告时间": pd.Timestamp("2026-09-17"),
         "公告链接": "implementation"},
    ])
    monkeypatch.setattr(ashare, "_announcement_frame", lambda *args: df)
    monkeypatch.setattr(
        ashare, "_pdf_text_from_announcement",
        lambda url: "每10股派发现金股利人民币2.49元，合计派发48.32亿元。"
        if url == "implementation" else "没有分红数据"
    )
    result = ashare._dividend_disclosure_evidence("000001.SZ", "2026-09-30")
    assert "2.49元" in result
    assert "48.32亿元" in result
    assert "implementation" in result
    assert "future" not in result


def test_fundamentals_includes_dividend_primary_source(monkeypatch):
    monkeypatch.setattr(ashare, "company_identity", lambda ticker: {"company_name": "平安银行"})
    monkeypatch.setattr(ashare, "_statement", lambda *args: "verified financial statement")
    monkeypatch.setattr(
        ashare, "_dividend_disclosure_evidence",
        lambda *args: "Official: every 10 shares 2.49 CNY; total 48.32 billion CNY",
    )
    result = ashare.get_fundamentals("000001.SZ", "2026-09-30")
    assert "2.49 CNY" in result
    assert result.count("verified financial statement") == 2
