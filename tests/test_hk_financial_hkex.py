"""Point-in-time HKEX/Eastmoney integration without external network requests."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import hk_financial, hkex

_BASE = "https://www1.hkexnews.hk/listedco/listconews/sehk/2026/"


@pytest.fixture
def hkex_data(monkeypatch):
    hkex._stock_id.cache_clear()
    hk_financial._filings_for_analysis.cache_clear()
    hk_financial._source_rows.cache_clear()
    calls = []
    news = [
        {
            "STOCK_CODE": "09988",
            "TITLE": "Alibaba Group Announces June Quarter 2026 Results",
            "DATE_TIME": "20/08/2026 17:46",
            "FILE_LINK": "/listedco/listconews/sehk/2026/0820/2026082000123.pdf",
            "LONG_TEXT": "Quarterly Results",
        },
        {
            "STOCK_CODE": "09988",
            "TITLE": "ANNUAL RESULTS FOR THE YEAR ENDED 31 MARCH 2026",
            "DATE_TIME": "15/05/2026 17:35",
            "FILE_LINK": "/listedco/listconews/sehk/2026/0515/2026051500999.pdf",
            "LONG_TEXT": "Final Results",
        },
        {
            "STOCK_CODE": "09988",
            "TITLE": "Alibaba Group Announces September Quarter 2026 Results",
            "DATE_TIME": "06/11/2026 17:35",
            "FILE_LINK": "/listedco/listconews/sehk/2026/1106/2026110600999.pdf",
        },
        {
            "STOCK_CODE": "00700",
            "TITLE": "Tencent June Quarter 2026 Results",
            "DATE_TIME": "20/08/2026 17:35",
            "FILE_LINK": "/listedco/listconews/sehk/2026/0820/2026082000777.pdf",
        },
        {
            "STOCK_CODE": "09988",
            "TITLE": "Notice of Board Meeting",
            "DATE_TIME": "01/08/2026 12:00",
            "FILE_LINK": "/listedco/listconews/sehk/2026/0801/2026080100123.pdf",
        },
        {
            "STOCK_CODE": "09988",
            "TITLE": "Undated quarterly results",
            "DATE_TIME": "not-a-timestamp",
            "FILE_LINK": "/listedco/listconews/sehk/2026/0801/2026080100888.pdf",
        },
    ]

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, params))
        if "activestock" in url:
            return Response({"stock": [{"c": "09988", "i": 1000015694},
                                       {"c": "00700", "i": 7609}]})
        return Response({"result": json.dumps(news), "hasNextRow": False})

    monkeypatch.setattr(hkex.requests, "get", get)
    yield calls
    hkex._stock_id.cache_clear()
    hk_financial._filings_for_analysis.cache_clear()
    hk_financial._source_rows.cache_clear()


@pytest.fixture
def financial_rows(monkeypatch, hkex_data):
    sample = [
        ("2026-06-30", "2026-04-01", "total_revenue", "营业总收入", 10000),
        ("2026-06-30", "2026-04-01", "net_income", "净利润", 1000),
        ("2026-09-30", "2026-07-01", "future_revenue", "未来季度收入", 999999),
        ("2026-03-31", "2025-04-01", "annual_revenue", "全年收入", 30000),
    ]
    table = pd.DataFrame([
        {
            "SECURITY_CODE": "09988",
            "REPORT_DATE": end,
            "START_DATE": start,
            "STD_ITEM_CODE": item,
            "STD_ITEM_NAME": name,
            "AMOUNT": value,
        }
        for end, start, item, name, value in sample
    ])
    seen = []

    def report(**kwargs):
        seen.append(kwargs)
        return table.copy()

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_financial_hk_report_em=report
    ))
    yield seen
    hk_financial._source_rows.cache_clear()


@pytest.mark.unit
def test_hkex_lookup_and_official_announcements_use_release_timestamp(hkex_data):
    rows = hkex.announcements("9988.HK", "2026-08-01", "2026-11-04")
    assert len(rows) == 2
    assert "June Quarter" in rows[0]["title"]
    assert rows[0]["published"].strftime("%Y-%m-%d %H:%M") == "2026-08-20 17:46"
    assert rows[0]["url"].startswith(_BASE)
    _, params = next((url, p) for url, p in hkex_data if "titleSearchServlet" in url)
    assert params["stockId"] == "1000015694"
    assert params["searchType"] == "0"
    assert params["toDate"] == "20261104"
    assert params["lang"] == "E"


@pytest.mark.unit
def test_hkex_news_tool_routing_uses_official_date_and_document(hkex_data):
    from tradingagents.dataflows.router import route_to_vendor

    out = route_to_vendor("get_news", "9988.HK", "2026-08-01", "2026-11-04")
    assert "2026-08-20 17:46" in out
    assert "hkexnews.hk" in out
    assert "September Quarter" not in out
    assert "Tencent" not in out


@pytest.mark.unit
def test_financial_reports_do_not_leak_before_or_on_publication_day(financial_rows):
    from tradingagents.dataflows.router import route_to_vendor

    for asof in ("2026-08-19", "2026-08-20"):
        out = route_to_vendor("get_income_statement", "9988.HK", "quarterly", asof)
        assert "NO_DATA_AVAILABLE" in out, out
        assert "10000" not in out


@pytest.mark.unit
def test_financial_reports_released_next_day_without_future_period(financial_rows):
    from tradingagents.dataflows.router import route_to_vendor

    out = route_to_vendor("get_income_statement", "9988.HK", "quarterly", "2026-11-04")
    assert "2026-06-30" in out
    assert "2026-08-20 17:46" in out
    assert "10000" in out
    assert "2026-09-30" not in out
    assert "999999" not in out
    assert "current database" in out
    assert financial_rows[0]["stock"] == "09988"
    assert financial_rows[0]["indicator"] == "报告期"


@pytest.mark.unit
def test_annual_period_requires_twelve_months(financial_rows):
    report = hk_financial.get_income_statement("9988.HK", "annual", "2026-11-04")
    assert "2026-03-31 (12 months)" in report
    assert "全年收入" in report
    assert "2026-06-30" not in report
    assert financial_rows[0]["indicator"] == "年度"


@pytest.mark.unit
def test_hk_fundamentals_combines_three_kinds(financial_rows):
    from tradingagents.dataflows.router import route_to_vendor

    out = route_to_vendor("get_fundamentals", "9988.HK", "2026-11-04")
    for name in ("资产负债表", "利润表", "现金流量表"):
        assert name in out
    assert "Original" not in out
    assert out.count("Official original filing:") == 3
    assert {call["symbol"] for call in financial_rows} == {"资产负债表", "利润表", "现金流量表"}


@pytest.mark.unit
def test_filing_with_unverifiable_period_never_unlocks_financials(monkeypatch, financial_rows):
    monkeypatch.setattr(hkex, "result_filings", lambda *args: [{
        "title": "Quarterly Results 2026",
        "published": pd.Timestamp("2026-08-20 17:46").to_pydatetime(),
        "url": _BASE + "0820/2026082000123.pdf",
    }])
    monkeypatch.setattr(hk_financial, "_pdf_first_pages", lambda url: "Unrelated text")
    hk_financial._filings_for_analysis.cache_clear()
    with pytest.raises(NoMarketDataError, match="no income report"):
        hk_financial.get_income_statement("9988.HK", "quarterly", "2026-11-04")


@pytest.mark.unit
def test_pdf_period_text_can_verify_a_generic_results_title(monkeypatch, financial_rows):
    monkeypatch.setattr(hkex, "result_filings", lambda *args: [{
        "title": "Quarterly Results Announcement",
        "published": pd.Timestamp("2026-08-20 17:46").to_pydatetime(),
        "url": _BASE + "0820/2026082000123.pdf",
    }])
    monkeypatch.setattr(
        hk_financial, "_pdf_first_pages",
        lambda url: "UNAUDITED RESULTS FOR THE THREE MONTHS ENDED 30 JUNE 2026",
    )
    hk_financial._filings_for_analysis.cache_clear()
    out = hk_financial.get_cashflow("9988.HK", "quarterly", "2026-11-04")
    assert "2026-06-30" in out
    assert "2026-08-20" in out


@pytest.mark.unit
def test_missing_or_malformed_hkex_json_fails_closed(monkeypatch):
    monkeypatch.setattr(hkex, "_fetch_json", lambda url, **kwargs: {"wrong": []})
    hkex._stock_id.cache_clear()
    with pytest.raises(VendorUnavailableError, match="unknown format"):
        hkex.announcements("9988.HK", "2026-08-01", "2026-11-04")
    with pytest.raises(VendorUnavailableError, match="result field"):
        hkex._result_rows({"error": "bad"})


@pytest.mark.unit
def test_no_report_period_or_bad_ticker_is_not_fabricated(hkex_data):
    assert hk_financial._period_in_text("Announcement of 2026 Results") is None
    assert hk_financial._period_in_text("June Quarter 2026 Results") == (
        pd.Timestamp("2026-06-30").date(), 3
    )
    assert hk_financial._period_in_text("FOR THE YEAR ENDED 31 MARCH 2026") == (
        pd.Timestamp("2026-03-31").date(), 12
    )
    with pytest.raises(ValueError, match="Hong Kong"):
        hkex.announcements("BABA", "2026-08-01", "2026-11-04")


@pytest.fixture
def balance_without_start_date(monkeypatch, hkex_data):
    """Real HK balance-sheet API omits START_DATE entirely."""
    table = pd.DataFrame([
        {"SECURITY_CODE": "09988", "REPORT_DATE": end,
         "STD_ITEM_CODE": item, "STD_ITEM_NAME": label, "AMOUNT": value}
        for end, item, label, value in [
            ("2026-06-30", "assets", "资产总额", 123400),
            ("2026-06-30", "liabilities", "负债总额", 11100),
            ("2026-09-30", "future", "未来资产", 999999),
            ("2026-03-31", "annual", "年度资产", 567890),
        ]
    ])
    calls = []

    def report(**kwargs):
        calls.append(kwargs)
        assert kwargs["symbol"] == "资产负债表"
        return table.copy()

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_financial_hk_report_em=report),
    )
    hk_financial._source_rows.cache_clear()
    yield calls
    hk_financial._source_rows.cache_clear()


@pytest.mark.unit
def test_hk_balance_sheet_without_start_date_matches_official_period(balance_without_start_date):
    from tradingagents.dataflows.router import route_to_vendor

    out = route_to_vendor("get_balance_sheet", "9988.HK", "quarterly", "2026-11-04")
    assert "Balance sheet as of 2026-06-30" in out
    assert "disclosed with 3-month results" in out
    assert "2026-08-20 17:46" in out
    assert "资产总额" in out and "123400" in out
    assert "2026-09-30" not in out
    assert "999999" not in out
    assert "2026-03-31" not in out
    assert balance_without_start_date[0]["indicator"] == "报告期"


@pytest.mark.unit
def test_hk_balance_sheet_without_start_date_is_unavailable_before_release(
    balance_without_start_date,
):
    from tradingagents.dataflows.router import route_to_vendor

    for asof in ("2026-08-19", "2026-08-20"):
        out = route_to_vendor("get_balance_sheet", "9988.HK", "quarterly", asof)
        assert out.startswith("NO_DATA_AVAILABLE:"), out
        assert "123400" not in out


@pytest.mark.unit
def test_hk_annual_balance_sheet_without_start_date(balance_without_start_date):
    out = hk_financial.get_balance_sheet("9988.HK", "annual", "2026-11-04")
    assert "Balance sheet as of 2026-03-31" in out
    assert "disclosed with 12-month results" in out
    assert "年度资产" in out
    assert "2026-06-30" not in out
    assert balance_without_start_date[0]["indicator"] == "年度"


@pytest.mark.unit
def test_hk_balance_sheet_requires_unambiguous_official_period(
    monkeypatch, balance_without_start_date,
):
    monkeypatch.setattr(hkex, "result_filings", lambda *args: [{
        "title": "Financial results for the period ended 30 June 2026",
        "published": pd.Timestamp("2026-08-20 17:46").to_pydatetime(),
        "url": _BASE + "0820/2026082000123.pdf",
    }])
    monkeypatch.setattr(hk_financial, "_pdf_first_pages", lambda url: "")
    hk_financial._filings_for_analysis.cache_clear()
    with pytest.raises(NoMarketDataError, match="no balance report"):
        hk_financial.get_balance_sheet("9988.HK", "quarterly", "2026-11-04")


@pytest.mark.unit
def test_hk_income_statement_still_requires_start_date(
    monkeypatch, balance_without_start_date,
):
    import sys

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_financial_hk_report_em=lambda **kwargs: pd.DataFrame([{
            "SECURITY_CODE": "09988", "REPORT_DATE": "2026-06-30",
            "STD_ITEM_CODE": "revenue", "STD_ITEM_NAME": "营业收入",
            "AMOUNT": 100,
        }]),
    ))
    hk_financial._source_rows.cache_clear()
    with pytest.raises(VendorUnavailableError, match="START_DATE"):
        hk_financial.get_income_statement("9988.HK", "quarterly", "2026-11-04")
