"""SEC EDGAR 财报回归测试：使用历史实际披露版本，而非今天的最新重述值。

其他提供商可能只按财务期末过滤今天的数字，
导致历史运行提前看到尚未申报或后来重述的报表。
EDGAR 逐项保存提交日期，可以按运行日期限制可见内容。"""

from __future__ import annotations

from unittest import mock

import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import sec_edgar

_REAL_FETCH = sec_edgar._fetch_json

TICKER_MAP = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


def _fact(end, val, filed, form="10-K", fp="FY", start=None):
    fact = {"end": end, "val": val, "filed": filed, "form": form, "fy": int(end[:4]), "fp": fp}
    if start:
        fact["start"] = start
    return fact


FACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [
            _fact("2008-09-27", 39_572_000_000, "2008-11-05"),
            _fact("2008-09-27", 36_171_000_000, "2010-01-25", form="10-K/A"),
            _fact("2022-03-26", 350_662_000_000, "2022-04-29", form="10-Q", fp="Q2"),
            _fact("2024-09-28", 364_980_000_000, "2024-11-01"),
        ]}},
        "Liabilities": {"units": {"USD": [_fact("2024-09-28", 308_030_000_000, "2024-11-01")]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            _fact("2024-09-28", 6.08, "2024-11-01", start="2023-09-30"),
        ]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            # 一份申报可在同一期末同时包含单季度和年初至今累计数值。
            _fact("2025-12-31", 81_300_000_000, "2026-01-29", form="10-Q", fp="Q2", start="2025-10-01"),
            _fact("2025-12-31", 158_900_000_000, "2026-01-29", form="10-Q", fp="Q2", start="2025-07-01"),
            _fact("2024-09-28", 391_035_000_000, "2024-11-01", start="2023-09-30"),
        ]}},
    }},
}


@pytest.fixture(autouse=True)
def _no_network_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sec_edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(sec_edgar, "_fetch_json", lambda url: TICKER_MAP if "company_tickers" in url else FACTS)


@pytest.mark.unit
def test_a_us_filer_resolves_to_its_cik():
    assert sec_edgar.cik_for("AAPL") == "0000320193"
    assert sec_edgar.cik_for("aapl") == "0000320193"


@pytest.mark.unit
def test_a_non_filer_is_reported_as_such_not_as_missing_data():
    with pytest.raises(NoMarketDataError, match="not a US SEC filer"):
        sec_edgar.get_balance_sheet("0700.HK", "annual", "2026-01-01")


@pytest.mark.unit
def test_a_restated_figure_reads_as_it_did_at_the_time():
    """使用当时已披露的数字，不能提前看到后来的更正。"""
    as_filed = sec_edgar.get_balance_sheet("AAPL", "annual", "2009-06-30")
    restated = sec_edgar.get_balance_sheet("AAPL", "annual", "2011-01-01")
    assert "39572" in as_filed and "36171" not in as_filed
    assert "36171" in restated


@pytest.mark.unit
def test_a_period_that_ended_but_was_not_filed_yet_is_not_served():
    """财政年度在 2024-09-28 结束，但数据到 2024-11-01 才公开。"""
    before = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-10-15")
    after = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    assert "2024-09-28" not in before
    assert "2024-09-28" in after and "364980" in after


@pytest.mark.unit
def test_the_quarter_is_not_confused_with_the_year_to_date():
    """同一份申报可能在相同期末包含不同跨度的财务数据。"""
    out = sec_edgar.get_income_statement("AAPL", "quarterly", "2026-06-01")
    assert "81300" in out
    assert "158900" not in out


@pytest.mark.unit
def test_a_line_the_filer_does_not_tag_is_named_unavailable():
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    assert "Stockholders Equity" in out and "unavailable" in out
    assert "364980" in out  # the rest of the statement still returns


@pytest.mark.unit
def test_the_report_states_the_vintage_rule():
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    assert "filed on or before 2024-11-15" in out


@pytest.mark.unit
def test_a_filer_with_no_usable_facts_reads_differently_from_a_non_filer(monkeypatch):
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else {"facts": {}})
    with pytest.raises(NoMarketDataError, match="no us-gaap facts"):
        sec_edgar.get_balance_sheet("AAPL", "annual", "2026-01-01")


@pytest.mark.unit
def test_company_facts_are_fetched_once_per_company_not_once_per_date(tmp_path, monkeypatch):
    """批量扫描多个日期时，申报历史本身保持相同文件。"""
    calls = []
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: calls.append(url) or (TICKER_MAP if "company_tickers" in url else FACTS))
    for date in ("2024-11-15", "2025-01-15", "2025-06-15"):
        sec_edgar.get_balance_sheet("AAPL", "annual", date)
    assert len([u for u in calls if "companyfacts" in u]) == 1


@pytest.mark.unit
def test_it_works_unconfigured_and_takes_the_caller_s_own_contact(monkeypatch):
    """SEC 对缺少联系方式的 User-Agent 可能返回 403；
    默认标识应包含联系信息，用户自定义值应覆盖它。"""
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    assert "@" in sec_edgar._user_agent()

    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "MyDesk research@example.com")
    assert sec_edgar._user_agent() == "MyDesk research@example.com"


@pytest.mark.unit
def test_a_throttle_lets_the_next_vendor_try(monkeypatch):
    """SEC 拒绝限流请求时，路由器应回退尝试 yfinance。"""
    import requests

    from tradingagents.dataflows.errors import VendorUnavailableError

    def _throttled(*a, **k):
        raise requests.HTTPError(response=mock.Mock(status_code=429))

    monkeypatch.setattr(sec_edgar.requests, "get", _throttled)
    with pytest.raises(VendorUnavailableError):
        _REAL_FETCH("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json")


@pytest.mark.unit
def test_values_do_not_break_the_columns():
    """金额可能达到数十亿；千位分隔符会破坏 CSV 字段结构。"""
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    body = [row for row in out.splitlines() if row.startswith("Total Assets")][0]
    assert body.count(",") == out.splitlines()[3].count(",")


@pytest.mark.unit
def test_a_per_share_figure_keeps_its_own_unit():
    """多数财务报表以百万为单位，但 EPS 使用美元/股，
    不能按金额同等缩放，否则真实每股收益可能显示为零。"""
    out = sec_edgar.get_income_statement("AAPL", "annual", "2024-11-15")
    row = [r for r in out.splitlines() if r.startswith("Diluted EPS")][0]
    assert "6.08" in row
    assert "USD/shares" in row or "per share" in row


@pytest.mark.unit
def test_every_row_has_one_cell_per_period():
    """没有匹配财务标签的行仍需保持完整列数，避免表格位置错乱。"""
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    table = [r for r in out.splitlines() if r and not r.startswith("#")]
    widths = {row.count(",") for row in table}
    assert len(widths) == 1, table


@pytest.mark.unit
def test_a_server_error_lets_the_next_vendor_try(monkeypatch):
    import requests

    from tradingagents.dataflows.errors import VendorError

    def _server_error(*a, **k):
        raise requests.HTTPError(response=mock.Mock(status_code=503))

    monkeypatch.setattr(sec_edgar.requests, "get", _server_error)
    with pytest.raises(VendorError):  # not a bare HTTPError
        _REAL_FETCH("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json")


@pytest.mark.unit
def test_older_periods_fall_back_to_the_tag_the_filer_used_then():
    """财务准则变化后，公司可能更改申报标签；
    每个时期选取一个有值的标签，不能把不同标签加总重复计数。"""
    facts = {"Revenues": {"units": {"USD": [_fact("2015-09-26", 233_715_000_000, "2015-10-28",
                                                 start="2014-09-28")]}},
             "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                 _fact("2024-09-28", 391_035_000_000, "2024-11-01", start="2023-09-30")]}}}
    values, unit = sec_edgar._as_of(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax",
                                            "Revenues"), "2026-01-01", ((300, 400),))
    assert values == {("2015-09-26", 0): 233_715_000_000, ("2024-09-28", 0): 391_035_000_000}
    assert unit == "USD"


@pytest.mark.unit
def test_a_period_reported_under_two_tags_takes_the_preferred_one_not_both():
    facts = {"Revenues": {"units": {"USD": [_fact("2024-09-28", 111, "2024-11-01", start="2023-09-30")]}},
             "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                 _fact("2024-09-28", 999, "2024-11-01", start="2023-09-30")]}}}
    values, _ = sec_edgar._as_of(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax",
                                         "Revenues"), "2026-01-01", ((300, 400),))
    assert values == {("2024-09-28", 0): 999}


@pytest.mark.unit
def test_the_default_identification_names_the_package_version(monkeypatch):
    """SEC 要求自动化客户端提供身份标识，版本信息应来自实际软件版本。"""
    import tradingagents

    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    assert sec_edgar._user_agent() == f"TradingAgents/{tradingagents.__version__} (contact@example.com)"


@pytest.mark.unit
def test_capital_expenditure_is_found_under_either_tag_filers_use(monkeypatch):
    """NVIDIA 和 Amazon 可能将资本性支出标记为生产性资产购买，
    而非字面上的不动产及设备购买。"""
    facts = {"facts": {"us-gaap": {"PaymentsToAcquireProductiveAssets": {"units": {"USD": [
        _fact("2024-12-31", 70_000_000, "2025-02-10", start="2024-01-01")]}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)
    out = sec_edgar.get_cashflow("AAPL", "annual", "2025-03-01")
    assert [r for r in out.splitlines() if r.startswith("Capital Expenditure")] == ["Capital Expenditure,70"]


def _columns(out):
    return [line for line in out.splitlines() if line.startswith(",")][0].split(",")[1:]


@pytest.mark.unit
def test_an_annual_balance_sheet_has_no_quarter_end_columns():
    """资产负债表是时点数据，没有期间跨度；
    不能因此把 10-Q 的季度末余额误认为年度报表。"""
    annual = _columns(sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15"))
    quarterly = _columns(sec_edgar.get_balance_sheet("AAPL", "quarterly", "2024-11-15"))
    assert "2022-03-26" not in annual and "2024-09-28" in annual
    assert "2022-03-26" in quarterly


@pytest.mark.unit
def test_a_twelve_month_total_from_a_quarterly_report_is_not_a_fiscal_year(monkeypatch):
    """Amazon 的 10-Q 可能披露过去十二个月累计值，
    即使跨度符合一年，也不能错误当成年报并与真实财年重叠。"""
    facts = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        _fact("2024-12-31", 115_000_000_000, "2025-02-07", start="2024-01-01"),
        _fact("2025-03-31", 113_000_000_000, "2025-05-02", form="10-Q", fp="Q1", start="2024-04-01"),
    ]}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)
    assert _columns(sec_edgar.get_cashflow("AAPL", "annual", "2025-06-01")) == ["2024-12-31"]


@pytest.mark.unit
def test_a_recast_outside_the_annual_report_still_counts_from_its_filing(monkeypatch):
    """拆分或剥离后，公司可能通过 8-K 重述历史年度；
    年度报告决定展示哪些列，而数值应取该历史日期之前最新的有效申报。"""
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
        _fact("2017-03-31", 16.97, "2017-06-15", form="20-F", start="2016-04-01"),
        _fact("2017-03-31", 2.12, "2019-09-30", form="6-K", start="2016-04-01"),
    ]}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)

    def eps(date):
        out = sec_edgar.get_income_statement("AAPL", "annual", date)
        return [line for line in out.splitlines() if line.startswith("Diluted EPS")][0].split(",")[1]

    assert eps("2019-01-01") == "16.97"
    assert eps("2020-01-01") == "2.12"


@pytest.mark.unit
def test_a_cash_flow_filed_year_to_date_is_served_with_its_span(monkeypatch):
    """许多申报人只在 10-Q 中提供年初至今的现金流累计数
    （3、6、9 个月）；不能只依赖独立季度值，否则每年第一季之后会缺数据。"""
    ytd = [
        _fact("2025-12-27", 30_000_000_000, "2026-01-30", form="10-Q", fp="Q1", start="2025-09-28"),
        _fact("2026-03-28", 53_000_000_000, "2026-05-01", form="10-Q", fp="Q2", start="2025-09-28"),
        _fact("2026-06-27", 81_000_000_000, "2026-08-01", form="10-Q", fp="Q3", start="2025-09-28"),
    ]
    facts = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ytd}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)

    out = sec_edgar.get_cashflow("AAPL", "quarterly", "2026-09-28")

    assert _columns(out) == ["2025-12-27", "2026-03-28 (6 months)", "2026-06-27 (9 months)"]
    assert [r for r in out.splitlines() if r.startswith("Operating Cash Flow")] == ["Operating Cash Flow,30000,53000,81000"]


@pytest.mark.unit
def test_a_row_filed_only_year_to_date_keeps_its_figure_beside_a_row_filed_by_quarter(monkeypatch):
    facts = {"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            _fact("2026-06-30", 120_000_000, "2026-08-01", form="10-Q", fp="Q2", start="2026-01-01")]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            _fact("2026-06-30", 10_000_000, "2026-08-01", form="10-Q", fp="Q2", start="2026-04-01")]}},
    }}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)

    out = sec_edgar.get_cashflow("AAPL", "quarterly", "2026-09-28")

    assert _columns(out) == ["2026-06-30", "2026-06-30 (6 months)"]
    rows = {r.split(",")[0]: r.split(",")[1:] for r in out.splitlines()[3:]}
    assert rows["Operating Cash Flow"] == ["", "120"]
    assert rows["Capital Expenditure"] == ["10", ""]
