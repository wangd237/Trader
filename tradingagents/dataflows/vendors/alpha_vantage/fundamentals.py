from tradingagents.dataflows.date_window import withhold_live_profile, withhold_undated_statements
from tradingagents.dataflows.vendors.alpha_vantage.common import _make_api_request


def get_fundamentals(ticker: str, as_of_date: str = None) -> str:
    """使用 Alpha Vantage 获取指定 ticker 的完整基本面数据。

    OVERVIEW 只提供当前值，没有历史数据版本，因此历史 ``as_of_date``
    必须拒绝返回，以免回测泄漏决策后的数据（#1300）。
    下方财报接口也没有实际披露日期，遵守相同约束。

    参数：
        ticker (str)：公司交易代码。
        as_of_date (str)：分析日期，格式 yyyy-mm-dd。

    返回：
        str：包含财务比率与核心指标的公司概况。"""
    withheld = withhold_live_profile(as_of_date, ticker)
    if withheld:
        return withheld

    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str = None):
    """使用 Alpha Vantage 获取指定 ticker 的资产负债表。"""
    withheld = withhold_undated_statements(as_of_date, ticker, "Balance Sheet")
    if withheld:
        return withheld
    return _make_api_request("BALANCE_SHEET", {"symbol": ticker})


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str = None):
    """使用 Alpha Vantage 获取指定 ticker 的现金流量表。"""
    withheld = withhold_undated_statements(as_of_date, ticker, "Cash Flow")
    if withheld:
        return withheld
    return _make_api_request("CASH_FLOW", {"symbol": ticker})


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str = None):
    """使用 Alpha Vantage 获取指定 ticker 的利润表。"""
    withheld = withhold_undated_statements(as_of_date, ticker, "Income Statement")
    if withheld:
        return withheld
    return _make_api_request("INCOME_STATEMENT", {"symbol": ticker})

