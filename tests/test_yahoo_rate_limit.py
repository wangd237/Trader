"""Yahoo 限流应先退避重试，耗尽后明确报告限流。

如果模型收到“股票可能无效或退市”，就会把未验证的提供商故障
误当作公司事实；但工具异常也不能直接终止运行。"""

import pandas as pd
import pytest
import yfinance as yf
from yfinance.data import YfData
from yfinance.exceptions import YFRateLimitError

from tradingagents.agents.tools import (
    get_global_news,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo import common, fundamentals, ohlcv

DAY = "2026-09-18"


def _rate_limited(*args, **kwargs):
    raise YFRateLimitError()


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(common.time, "sleep", lambda seconds: None)
    for module in (common, fundamentals):
        monkeypatch.setattr(module, "vendor_reachable", lambda url: True)
    return monkeypatch


@pytest.mark.unit
@pytest.mark.parametrize("tool, args", [
    pytest.param(get_stock_data, ("AAPL", "2026-09-10", DAY), id="stock_data"),
    pytest.param(get_indicators, ("AAPL", "rsi", DAY, 5), id="indicators"),
    pytest.param(get_verified_market_snapshot, ("AAPL", DAY), id="snapshot"),
    pytest.param(get_news, ("AAPL", "2026-09-10", DAY), id="news"),
    pytest.param(get_global_news, (DAY, 7, 5), id="global_news"),
    pytest.param(get_insider_transactions, ("AAPL",), id="insider"),
])
def test_a_rate_limit_that_outlasts_the_retries_is_reported_as_one(yahoo, tool, args):
    for name in ("history", "get_news"):
        yahoo.setattr(yf.Ticker, name, _rate_limited)
    yahoo.setattr(yf.Ticker, "insider_transactions", property(_rate_limited))
    yahoo.setattr(yf, "Search", _rate_limited)

    # 本例为当日调用；历史运行根本不应返回没有披露日期的内部人交易。
    out = tool.func(*args, trade_date=get_current_date())

    assert out.startswith("DATA_UNAVAILABLE"), out
    assert "delisted" not in out


@pytest.mark.unit
def test_the_indicator_path_retries_a_rate_limit(yahoo):
    """技术指标行情通过 yf.download 获取；该函数对 HTTP 429 返回空表，
    导致旧实现不会重试。"""
    bar = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0],
                        "Volume": [100]}, index=pd.DatetimeIndex([DAY], name="Date"))
    answers = [YFRateLimitError(), bar]

    def history(self, **kwargs):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    yahoo.setattr(yf.Ticker, "history", history)

    assert ohlcv.load_ohlcv("AAPL", DAY)["Close"].tolist() == [1.0]


@pytest.mark.unit
def test_yfinance_raises_the_rate_limit_from_history(yahoo):
    yahoo.setattr(YfData, "_make_request", _rate_limited)

    with pytest.raises(VendorUnavailableError, match="rate limited"):
        ohlcv.load_ohlcv("AAPL", DAY)


@pytest.mark.unit
def test_fundamentals_ask_a_new_ticker_after_a_rate_limit(yahoo):
    """Ticker 会记住曾经失败的 info 请求；重复使用同一实例可能持续返回空值，
    看起来像是标的确实无数据。"""

    class Ticker:
        def __init__(self, symbol):
            self.fetched = False

        @property
        def info(self):
            if self.fetched:
                return {}
            self.fetched = True
            raise YFRateLimitError()

    yahoo.setattr(yf, "Ticker", Ticker)

    out = router.route_to_vendor("get_fundamentals", "AAPL", None)

    assert out.startswith("DATA_UNAVAILABLE"), out


@pytest.mark.unit
def test_a_price_request_that_raises_is_unavailable_whether_or_not_yahoo_answers(yahoo):
    """Ticker.history 对不存在的代码返回空表；
    真正抛异常时只能证明请求传输失败，不能推断代码不存在。"""

    def refused(self, **kwargs):
        raise ConnectionError("curl: (7) Failed to connect to query2.finance.yahoo.com")

    yahoo.setattr(yf.Ticker, "history", refused)
    for reachable in (False, True):
        yahoo.setattr(common, "vendor_reachable", lambda url, _r=reachable: _r)
        with pytest.raises(VendorUnavailableError, match="request failed"):
            ohlcv.load_ohlcv("AAPL", DAY)


@pytest.mark.unit
def test_a_price_request_that_fails_is_unavailable_even_when_yahoo_answers(monkeypatch, tmp_path):
    """超时并不能说明代码情况，主机可达也不等于该请求正常完成。"""
    def timed_out(**kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(ohlcv, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(ohlcv.yf, "Ticker", lambda s: type("T", (), {"history": lambda self, **k: timed_out(**k)})())
    monkeypatch.setattr(common, "vendor_reachable", lambda url: True)

    with pytest.raises(VendorUnavailableError):
        ohlcv.load_ohlcv("NVDA", "2026-09-23")


class _Unreachable:
    """模拟所有请求都在传输过程中失败的 Ticker。"""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        raise TimeoutError("read timed out")


def _yahoo_calls():
    from tradingagents.dataflows.date_window import get_current_date
    from tradingagents.dataflows.vendors.yahoo import market, news

    today = get_current_date()
    return [
        pytest.param(lambda: fundamentals.get_fundamentals("AAPL", today), id="fundamentals"),
        pytest.param(lambda: fundamentals.get_balance_sheet("AAPL", "quarterly", today), id="statement"),
        pytest.param(lambda: fundamentals.get_insider_transactions("AAPL", today), id="insider"),
        pytest.param(lambda: fundamentals.get_company_profile("AAPL"), id="profile"),
        pytest.param(lambda: news.get_news_yfinance("AAPL", "2026-09-10", DAY), id="news"),
        pytest.param(lambda: news.get_global_news_yfinance(DAY, 7, 5), id="global_news"),
        pytest.param(lambda: market.get_YFin_data_online("AAPL", "2026-09-10", DAY), id="stock_data"),
        pytest.param(lambda: market.get_closes("AAPL", "2026-09-10", DAY), id="closes"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("call", _yahoo_calls())
def test_a_yahoo_request_that_fails_is_unavailable_not_absent(yahoo, call):
    """无效代码通常返回空值；抛异常说明调用过程失败。"""
    yahoo.setattr(yf, "Ticker", _Unreachable)
    yahoo.setattr(yf, "Search", _Unreachable)

    with pytest.raises(VendorUnavailableError):
        call()


@pytest.mark.unit
@pytest.mark.parametrize("frame", [
    pytest.param(pd.DataFrame(), id="unknown_symbol"),
    pytest.param(pd.DataFrame({"Date": pd.to_datetime(["2026-09-25"]), "Open": [1.0], "High": [1.0],
                               "Low": [1.0], "Close": [1.0], "Volume": [1]}), id="no_row_by_the_date"),
])
def test_a_snapshot_with_no_rows_is_reported_not_raised(yahoo, frame):
    """工具未处理异常会终止运行；真正正常返回空结果才可能代表标的无数据。"""
    from tradingagents.dataflows.errors import NoMarketDataError
    from tradingagents.dataflows.vendors.yahoo import snapshot

    def load(symbol, as_of_date, **kwargs):
        if frame.empty:
            raise NoMarketDataError(symbol, symbol, "no price rows")
        return frame

    yahoo.setattr(snapshot, "load_ohlcv", load)

    out = get_verified_market_snapshot.func("ZZZZ", DAY, trade_date=DAY)

    assert out.startswith("NO_DATA_AVAILABLE"), out


def _yahoo_request_raises(yahoo, exc):
    """模拟 yfinance 内部所有请求抛出 ``exc``。"""
    from yfinance.data import YfData

    def request(self, *args, **kwargs):
        raise exc

    yahoo.setattr(YfData, "_make_request", request)


@pytest.mark.unit
@pytest.mark.parametrize("call", _yahoo_calls())
def test_a_request_that_yfinance_would_swallow_is_unavailable(yahoo, call):
    """yfinance 有时将失败转换成空结果；
    若 Yahoo 主机可达，仍不能贸然把空财报解释成真实无数据。"""
    _yahoo_request_raises(yahoo, TimeoutError("read timed out"))

    with pytest.raises(VendorUnavailableError):
        call()


@pytest.mark.unit
@pytest.mark.parametrize("call", [c for c in _yahoo_calls() if c.id in ("fundamentals", "stock_data", "insider")])
def test_yahoo_answering_not_found_is_no_data(yahoo, call):
    """Yahoo 对未知代码返回 HTTP 404，这是标的不存在的响应，不是服务故障。"""
    from curl_cffi.requests import Response
    from curl_cffi.requests.exceptions import HTTPError

    from tradingagents.dataflows.errors import NoMarketDataError

    not_found = Response()
    not_found.status_code = 404
    _yahoo_request_raises(yahoo, HTTPError("HTTP Error 404: ", response=not_found))

    try:
        out = call()
    except NoMarketDataError:
        return
    assert out.startswith("No insider transactions"), out   # an empty filing list is not an error


@pytest.mark.unit
@pytest.mark.parametrize("debug, reason, absent", [
    pytest.param(" (1d 2026-09-26 -> 2026-09-27)", None, True, id="a_chart_with_no_prices"),
    pytest.param(" (1d 1970-01-01 -> 1970-02-01)", "Data doesn't exist for startDate = 18000, endDate = 2696400",
                 True, id="yahoo_says_the_data_does_not_exist"),
    pytest.param(" (1d 2026-09-21 -> 2026-09-25)(Yahoo status_code = 503)", None, False, id="an_http_error"),
    pytest.param(" (1d 2026-09-21 -> 2026-09-25)", "Service temporarily unavailable", False,
                 id="yahoo_reports_a_service_failure"),
])
def test_only_yahoo_saying_a_window_has_no_prices_is_no_data(yahoo, debug, reason, absent):
    """YFPricesMissingError 既可能表示 Yahoo 明确无价格，也可能包含请求错误；
    只有前者可视为关于标的的真实结论。"""
    from yfinance.exceptions import YFPricesMissingError

    from tradingagents.dataflows.errors import NoMarketDataError
    from tradingagents.dataflows.vendors.yahoo import market

    def no_prices(self, **kwargs):
        raise YFPricesMissingError("AAPL", debug, yahoo_reason=reason)

    yahoo.setattr(yf.Ticker, "history", no_prices)

    with pytest.raises(NoMarketDataError if absent else VendorUnavailableError):
        market.get_YFin_data_online("AAPL", "2026-09-21", "2026-09-25")


@pytest.mark.unit
def test_a_missing_time_zone_is_not_taken_for_an_answer(yahoo):
    """时区查询失败与代码确实无时区的错误形式可能相同，不能混淆。"""
    from yfinance.exceptions import YFTzMissingError

    from tradingagents.dataflows.vendors.yahoo import market

    def no_tz(self, **kwargs):
        raise YFTzMissingError("AAPL")

    yahoo.setattr(yf.Ticker, "history", no_tz)

    with pytest.raises(VendorUnavailableError):
        market.get_YFin_data_online("AAPL", "2026-09-21", "2026-09-25")
