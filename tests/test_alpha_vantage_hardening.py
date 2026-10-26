"""Alpha Vantage 请求健壮性测试。

回归验证 #990（未设置网络超时会卡住）、#991（错误密钥被误判为限流）、
#1115（基本面内容是 JSON 字符串而非字典，导致历史过滤未执行），
以及历史运行必须丢弃结束日期之后行情的数据裁剪。"""
import json

import pytest

import tradingagents.dataflows.net as net
import tradingagents.dataflows.vendors.alpha_vantage.common as av
import tradingagents.dataflows.vendors.alpha_vantage.fundamentals as avf
import tradingagents.dataflows.vendors.alpha_vantage.stock as avs


class _FakeResponse:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _patched_get(body, capture=None):
    def fake_get(url, params=None, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        return _FakeResponse(body)
    return fake_get


@pytest.mark.unit
def test_request_passes_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(net.requests, "get", _patched_get("Date,Close\n2025-01-02,1.0", captured))
    av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert captured.get("timeout") == av.REQUEST_TIMEOUT  # #990


@pytest.mark.unit
def test_rate_limit_detected(monkeypatch):
    body = '{"Information": "Our standard API rate limit is 25 requests per day. ... your API key ..."}'
    monkeypatch.setattr(net.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageRateLimitError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_invalid_key_not_mislabeled_as_rate_limit(monkeypatch):
    # 无效密钥提示含有“API key”，不能将其误判为临时限流，
    # 而应明确报告配置错误（#991）。
    body = ('{"Information": "the parameter apikey is invalid or missing. '
            'Please claim your free API key on (https://www.alphavantage.co/support/#api-key)."}')
    monkeypatch.setattr(net.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotConfiguredError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    with pytest.raises(av.AlphaVantageRateLimitError):  # sanity: rate-limit path still distinct
        monkeypatch.setattr(net.requests, "get", _patched_get('{"Note": "API call frequency is 5 calls per minute."}'))
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_a_rejected_request_is_unavailable_not_data(monkeypatch):
    """Alpha Vantage 对无效标的和错误请求都可能返回 {"Error Message": ...}，
    因此无法据此判断标的是否真实存在；
    如果当作数据返回，可能被解释为单行 CSV 或无数据（#1442）。"""
    from tradingagents.dataflows import router
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.errors import VendorUnavailableError

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k-secret")
    body = '{"Error Message": "Invalid API call. Please retry or visit the documentation for TIME_SERIES_DAILY."}'
    monkeypatch.setattr(net.requests, "get", _patched_get(body))

    with pytest.raises(VendorUnavailableError) as raised:
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "ZZZZ"})
    assert "k-secret" not in str(raised.value)

    set_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})
    out = router.route_to_vendor("get_stock_data", "ZZZZ", "2026-01-02", "2026-01-09")
    assert out.startswith("DATA_UNAVAILABLE"), out


@pytest.mark.unit
@pytest.mark.parametrize("body", [
    '{"Note": "Thank you for using key k-secret. API call frequency is 5 calls per minute."}',
    '{"Information": "the apikey k-secret is invalid or missing."}',
])
def test_the_key_never_reaches_an_error_message(monkeypatch, body):
    """Alpha Vantage 可能在提示中回显密钥；错误内容会进入日志和报告。"""
    from tradingagents.dataflows.errors import VendorError

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k-secret")
    monkeypatch.setattr(net.requests, "get", _patched_get(body))

    with pytest.raises(VendorError) as raised:
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert "k-secret" not in str(raised.value)


_FUNDAMENTALS_JSON = json.dumps({
    "symbol": "AAPL",
    "annualReports": [
        {"fiscalDateEnding": "2025-12-31", "totalAssets": "1"},   # future -> must drop
        {"fiscalDateEnding": "2023-12-31", "totalAssets": "2"},   # past   -> must keep
    ],
    "quarterlyReports": [
        {"fiscalDateEnding": "2024-06-30", "totalAssets": "3"},   # future -> must drop
        {"fiscalDateEnding": "2023-09-30", "totalAssets": "4"},   # past   -> must keep
    ],
})


@pytest.mark.unit
def test_fundamentals_no_curr_date_passes_through(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: _FUNDAMENTALS_JSON)
    assert avf.get_income_statement("AAPL") == _FUNDAMENTALS_JSON


# ---------------------------------------------------------------------------
# Date trim (see the rationale on the unguarded trim in alpha_vantage.common)
# ---------------------------------------------------------------------------

_DAILY_CSV = (
    "timestamp,open,high,low,close,volume\n"
    "2024-05-13,1,1,1,1,10\n"   # after end_date -> must never be served
    "2024-05-10,1,1,1,1,10\n"
    "2024-05-09,1,1,1,1,10\n"
)


@pytest.mark.unit
def test_stock_data_is_trimmed_to_the_requested_window(monkeypatch):
    monkeypatch.setattr(avs, "_make_api_request", lambda *a, **k: _DAILY_CSV)
    out = avs.get_stock("IBM", "2024-05-09", "2024-05-10")
    assert "2024-05-10" in out and "2024-05-09" in out
    assert "2024-05-13" not in out, "bar after end_date leaked into the window"


@pytest.mark.unit
def test_unparseable_body_is_never_served_untrimmed(monkeypatch):
    """裁剪过程不能吞掉异常并返回未筛选的全部数据，
    否则历史回测会读取 end_date 之后的价格。"""
    monkeypatch.setattr(avs, "_make_api_request",
                        lambda *a, **k: "timestamp,close\nnot-a-date,1\n")

    with pytest.raises(ValueError):
        avs.get_stock("IBM", "2024-05-09", "2024-05-10")


@pytest.mark.unit
def test_empty_body_still_passes_through(monkeypatch):
    monkeypatch.setattr(avs, "_make_api_request", lambda *a, **k: "")
    assert avs.get_stock("IBM", "2024-05-09", "2024-05-10") == ""


def test_request_error_message_carries_no_key(monkeypatch):
    # Alpha Vantage 也把密钥写在 URL 中（#1324）。
    import requests
    key = "AVKEY1234567890XYZ"
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", key)

    def boom(*a, **k):
        raise requests.Timeout(f"Read timed out. url: https://www.alphavantage.co/query?apikey={key}")

    monkeypatch.setattr(net.requests, "get", boom)
    with pytest.raises(requests.Timeout) as caught:
        av._make_api_request("OVERVIEW", {"symbol": "IBM"})
    assert key not in str(caught.value)


@pytest.mark.unit
def test_global_news_omitted_optionals_use_the_configured_defaults(monkeypatch):
    """工具未收到 look_back_days 或 limit 时会传入 None（#1326）。"""
    from tradingagents.dataflows.vendors.alpha_vantage import news as alpha_vantage_news

    monkeypatch.setattr(alpha_vantage_news, "get_config",
                        lambda: {"global_news_lookback_days": 3, "global_news_article_limit": 9})
    seen = {}
    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", lambda fn, params: seen.update(params) or "{}")

    alpha_vantage_news.get_global_news("2026-08-14", None, None)

    assert seen["time_from"].startswith("20260811") and seen["limit"] == "9"


@pytest.mark.unit
def test_the_news_window_includes_the_analysis_day(monkeypatch):
    """time_to 曾指向结束日期当天零点，导致分析日全天发布的新闻被遗漏；
    应与 yfinance 路径一样包含整个结束日期。"""
    from tradingagents.dataflows.vendors.alpha_vantage import news as alpha_vantage_news

    seen = {}
    monkeypatch.setattr(alpha_vantage_news, "_make_api_request",
                        lambda fn, params: seen.update(params) or "{}")

    alpha_vantage_news.get_news("AAPL", "2026-03-10", "2026-03-14")

    assert seen["time_from"] == "20260310T0000"
    assert seen["time_to"] == "20260314T2359"


@pytest.mark.unit
@pytest.mark.parametrize("indicator", ["vwma", "mfi"])
def test_an_indicator_this_vendor_lacks_lets_the_next_one_serve_it(indicator):
    """如果返回普通文字，路由器会误以为调用成功，
    导致当前数据源不支持指标时也无法回退到下一提供商。"""
    from tradingagents.dataflows.errors import VendorError
    from tradingagents.dataflows.vendors.alpha_vantage import indicator as alpha_vantage_indicator

    with pytest.raises(VendorError):
        alpha_vantage_indicator.get_indicator("AAPL", indicator, "2026-05-08", 30)


@pytest.mark.unit
def test_ticker_news_asks_for_only_as_many_articles_as_configured(monkeypatch):
    """接口默认返回 50 篇文章，且附带各文章的情绪数组；
    如果全部注入 Prompt，会增加大量上下文消耗。"""
    from tradingagents.dataflows.vendors.alpha_vantage import news as alpha_vantage_news

    monkeypatch.setattr(alpha_vantage_news, "get_config", lambda: {"news_article_limit": 8})
    seen = {}
    monkeypatch.setattr(alpha_vantage_news, "_make_api_request",
                        lambda fn, params: seen.update(params) or "{}")

    alpha_vantage_news.get_news("AAPL", "2026-03-10", "2026-03-14")

    assert seen["limit"] == "8"
