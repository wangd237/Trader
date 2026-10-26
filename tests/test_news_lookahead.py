"""yfinance 新闻不能向历史窗口泄漏未来文章，回测时也不能使用无日期文章。

回归 #992（扁平新闻绕过日期过滤）、#1007（全球新闻泄漏未来内容）、
#993（过滤为空后返回空正文）、#1126（结束日期次日零点被误纳入，
及宿主时区导致筛选结果不一致）。"""
from datetime import UTC, datetime

import pytest

import tradingagents.dataflows.vendors.yahoo.news as ynews
from tradingagents.dataflows.date_window import in_window


def _epoch(date_str):
    """将 ``date_str`` 的 UTC 零点转换为 Unix 秒数，不依赖宿主时区。"""
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())


@pytest.mark.unit
def test_flat_article_publish_time_is_parsed():
    # #992：扁平文章现在包含 pub_date；此前一直为 None，无法过滤。
    # #1126：日期按带时区 UTC 解析，避免随宿主时区偏移。
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-05-09")}
    )
    assert data["pub_date"] is not None
    assert data["pub_date"].tzinfo is not None
    assert data["pub_date"] == datetime(2025, 5, 9, tzinfo=UTC)


@pytest.mark.unit
def test_window_excludes_future_and_undated_in_backtest():
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)  # historical window (well in the past)
    inside = datetime(2025, 5, 5)
    future = datetime(2025, 6, 1)
    assert in_window(inside, start, end) is True
    assert in_window(future, start, end) is False     # look-ahead blocked
    assert in_window(None, start, end) is False        # undated -> excluded in backtest


@pytest.mark.unit
def test_window_keeps_undated_in_live_window():
    # 实时窗口延伸到今天，保留无日期文章。
    now = datetime.now(UTC)
    assert in_window(None, now, now) is True


@pytest.mark.unit
def test_upper_bound_is_exclusive():
    # #1126：结束日期次日零点的文章不能进入历史窗口；
    # 结束日期当天的文章仍应完整保留。
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)
    midnight_after = datetime(2025, 5, 10, 0, 0, 0, tzinfo=UTC)
    last_moment = datetime(2025, 5, 9, 23, 59, 59, tzinfo=UTC)
    assert in_window(midnight_after, start, end) is False
    assert in_window(last_moment, start, end) is True


@pytest.mark.unit
def test_offset_aware_timestamp_is_converted_not_truncated():
    # #1126：2025-05-10T01:00+05:00 实际对应 05-09 20:00Z，应在窗口内。
    # 直接去掉时区会错误地按 05-10 过滤掉。
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)
    aware = datetime.fromisoformat("2025-05-10T01:00:00+05:00")
    assert in_window(aware, start, end) is True


@pytest.mark.unit
def test_global_news_future_flat_article_excluded(monkeypatch):
    # #1007：未来日期的扁平全球新闻不能出现在历史运行中。
    future_article = {"title": "FUTURE EVENT", "publisher": "P", "link": "l",
                      "providerPublishTime": _epoch("2025-06-01")}
    past_article = {"title": "PAST EVENT", "publisher": "P", "link": "l",
                    "providerPublishTime": _epoch("2025-05-05")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [future_article, past_article]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "PAST EVENT" in out
    assert "FUTURE EVENT" not in out  # #1007


@pytest.mark.unit
def test_global_news_empty_after_filter_is_informative(monkeypatch):
    # #993：全部文章被过滤后，应明确说明原因，而不是返回空正文。
    only_future = {"title": "FUTURE", "publisher": "P", "link": "l",
                   "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [only_future]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "###" not in out  # no empty article body
    # 只返回了更晚的文章，说明该数据流未覆盖请求窗口。
    assert "unavailable" in out and "not an absence" in out


def _ticker_with(articles, monkeypatch, search_news=(), search_quotes=("AAPL",)):
    """模拟报价新闻流返回 ``articles``，搜索接口负责其余文章。
    默认搜索能识别 ticker，但没有相关新闻，表示 Yahoo 确实响应。"""
    class FakeTicker:
        def __init__(self, *a, **k):
            pass

        def get_news(self, count=20):
            return articles

    searched = []

    class FakeSearch:
        def __init__(self, query, **k):
            searched.append(query)
            self.news = list(search_news)
            self.quotes = [{"symbol": s} for s in search_quotes]

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    return searched


@pytest.mark.unit
def test_ticker_news_window_before_feed_coverage_is_unavailable(monkeypatch):
    # Yahoo 只提供近期文章，不能把历史窗口没有返回新闻
    # 当成该时期确实没有新闻。
    recent = [{"title": "RECENT", "publisher": "P", "link": "l",
               "providerPublishTime": _epoch("2026-09-10")}]
    _ticker_with(recent, monkeypatch)
    out = ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")
    assert "RECENT" not in out
    assert "unavailable" in out and "not an absence" in out
    assert "2026-09-10" not in out  # an article after the window


@pytest.mark.unit
def test_ticker_news_covered_but_empty_window_is_a_real_absence(monkeypatch):
    articles = [{"title": "RECENT", "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2026-09-10")},
                {"title": "OLDER", "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2026-07-01")}]
    _ticker_with(articles, monkeypatch)
    out = ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")
    assert "No news found" in out
    assert "unavailable" not in out


@pytest.mark.unit
@pytest.mark.parametrize("dates, expect_gap", [
    ([], True),                                              # empty feed: covers at most now
    ([None], True),                                          # undated only: same
    ([datetime(2026, 5, 20, tzinfo=UTC)], True),    # all after the window
    ([datetime(2026, 5, 4, tzinfo=UTC)], True),     # starts mid-window: partial
    ([datetime(2026, 5, 1, 18, tzinfo=UTC)], False),  # reaches the first day
    ([datetime(2026, 5, 20, tzinfo=UTC),
      datetime(2026, 4, 1, tzinfo=UTC)], False),    # coverage reaches back
])
def test_coverage_gap_boundaries(dates, expect_gap):
    from tradingagents.dataflows.date_window import coverage_gap

    out = coverage_gap(dates, "2026-05-01", "2026-05-08", "Feed", "items")
    assert (out is not None) is expect_gap
    if expect_gap:
        assert "unavailable for 2026-05-01..2026-05-08" in out and "not an absence" in out


@pytest.mark.unit
def test_ticker_news_empty_feed_for_a_past_window_is_unavailable(monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError

    _ticker_with([], monkeypatch)
    with pytest.raises(VendorUnavailableError):
        ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")


@pytest.mark.unit
def test_ticker_news_null_feed_is_handled(monkeypatch):
    # Yahoo 也可能返回 None 而非列表，应表示数据不可用而非程序崩溃。
    from tradingagents.dataflows.errors import VendorUnavailableError

    _ticker_with(None, monkeypatch)
    with pytest.raises(VendorUnavailableError):
        ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")


def _tagged(title, day, *tickers):
    return {"title": title, "publisher": "P", "link": "l",
            "providerPublishTime": _epoch(day), "relatedTickers": list(tickers)}


@pytest.mark.unit
def test_an_empty_quote_feed_falls_back_to_the_articles_search_tags_with_the_symbol(monkeypatch):
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    searched = _ticker_with([], monkeypatch, search_news=[
        _tagged("ABOUT AAPL", today, "AAPL", "MSFT"),
        _tagged("ABOUT GIS", today, "GIS"),
    ])
    out = ynews.get_news_yfinance("AAPL", today, today)
    assert searched == ["AAPL"]
    assert "ABOUT AAPL" in out and "ABOUT GIS" not in out


@pytest.mark.unit
def test_a_quote_feed_with_articles_is_not_searched(monkeypatch):
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    searched = _ticker_with([_tagged("FEED", today)], monkeypatch)
    out = ynews.get_news_yfinance("AAPL", today, today)
    assert "FEED" in out and searched == []


@pytest.mark.unit
def test_news_neither_yahoo_source_answers_is_a_vendor_outage(monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    _ticker_with([], monkeypatch, search_quotes=())
    with pytest.raises(VendorUnavailableError):
        ynews.get_news_yfinance("AAPL", today, today)


@pytest.mark.unit
def test_a_yahoo_news_outage_moves_to_the_next_configured_vendor(monkeypatch):
    from unittest.mock import patch

    from tradingagents.dataflows import router
    from tradingagents.dataflows.config import get_config, set_config

    _ticker_with([], monkeypatch, search_quotes=())
    calls = []

    def other_vendor(*a, **k):
        calls.append(a)
        return "OTHER VENDOR NEWS"

    saved = dict(get_config()["data_vendors"])
    set_config({"data_vendors": {**saved, "news_data": "yfinance,alpha_vantage"}})
    try:
        with patch.dict(router.VENDOR_METHODS, {"get_news": {
            "yfinance": ynews.get_news_yfinance, "alpha_vantage": other_vendor,
        }}):
            out = router.route_to_vendor("get_news", "AAPL", "2026-10-01", "2026-10-02")
    finally:
        set_config({"data_vendors": saved})
    assert out == "OTHER VENDOR NEWS" and len(calls) == 1


@pytest.mark.unit
def test_search_results_never_count_as_an_absence_of_news(monkeypatch):
    # 搜索结果只是按相关性抽取的稀疏样本，即使覆盖该窗口，
    # 也不能证明没有返回文章就等于当时没有新闻。
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    _ticker_with([], monkeypatch, search_news=[
        _tagged("NEWER", today, "AAPL"), _tagged("OLDER", "2026-01-05", "AAPL"),
    ])
    out = ynews.get_news_yfinance("AAPL", "2026-03-01", "2026-03-08")
    assert "No news found" not in out
    assert "unavailable" in out and "not an absence" in out
    assert "NEWER" not in out and "OLDER" not in out


@pytest.mark.unit
def test_a_symbol_yahoo_has_no_articles_for_is_left_to_the_next_vendor(monkeypatch):
    # 报价流异常时，Yahoo 未返回某标的的新闻
    # 并不能证明该公司没有相关新闻。
    from tradingagents.dataflows.errors import VendorUnavailableError

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    _ticker_with([], monkeypatch, search_quotes=("SHEL.L",),
                 search_news=[_tagged("ABOUT SHEL", today, "SHEL")])
    with pytest.raises(VendorUnavailableError, match="SHEL.L"):
        ynews.get_news_yfinance("SHEL.L", today, today)


@pytest.mark.unit
def test_search_tags_are_matched_whatever_their_case(monkeypatch):
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    _ticker_with([], monkeypatch, search_news=[_tagged("ABOUT AAPL", today, "aapl")])
    assert "ABOUT AAPL" in ynews.get_news_yfinance("AAPL", today, today)


@pytest.mark.unit
def test_a_search_with_articles_has_answered_even_without_listing_the_symbol(monkeypatch):
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    _ticker_with([], monkeypatch, search_quotes=(),
                 search_news=[_tagged("ABOUT AAPL", today, "AAPL")])
    assert "ABOUT AAPL" in ynews.get_news_yfinance("AAPL", today, today)


@pytest.mark.unit
def test_a_search_yahoo_answers_with_not_found_is_an_outage(monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError

    _ticker_with([], monkeypatch)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: [] if "get_news" in repr(fn.__code__.co_names) else None)
    with pytest.raises(VendorUnavailableError):
        ynews.get_news_yfinance("AAPL", "2026-10-01", "2026-10-02")



@pytest.mark.unit
def test_global_news_empty_feed_for_a_past_window_is_unavailable(monkeypatch):
    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = []

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_global_news_does_not_infer_coverage_from_a_stale_search_hit(monkeypatch):
    # 全球新闻合并了多次模糊搜索；窗口之前的一条旧文章
    # 不能证明中间每一天的历史内容已经覆盖。
    stale = {"title": "STALE", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-01-01")}
    fresh = {"title": "FRESH", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [fresh, stale]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "unavailable" in out and "No global news found" not in out


@pytest.mark.unit
def test_coverage_gap_future_window_is_unavailable():
    from datetime import timedelta

    from tradingagents.dataflows.date_window import coverage_gap
    today = datetime.now(UTC).date()
    out = coverage_gap([], str(today), str(today + timedelta(days=3)), "Feed", "items")
    assert out is not None and "past today" in out


@pytest.mark.unit
def test_out_of_window_articles_do_not_consume_the_article_budget(monkeypatch):
    """数量上限应计算本次运行允许读取的文章，而非抓取候选数（#1356）。
    以前窗口外文章占满预算，使窗口内文章被误报为不存在。"""
    stale = [{"title": f"OLD {i}", "publisher": "P", "link": "l",
              "providerPublishTime": _epoch("2025-01-01")} for i in range(2)]
    wanted = {"title": "IN WINDOW", "publisher": "P", "link": "l",
              "providerPublishTime": _epoch("2025-05-08")}
    pages = [stale, [wanted]]

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = pages.pop(0) if pages else []

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    monkeypatch.setattr(ynews, "get_config", lambda: {
        "global_news_lookback_days": 7, "global_news_article_limit": 2,
        "global_news_queries": ["markets", "economy"],
    })

    out = ynews.get_global_news_yfinance("2025-05-09")

    assert "IN WINDOW" in out
    assert "OLD 0" not in out


@pytest.mark.unit
def test_the_article_limit_still_caps_what_is_returned(monkeypatch):
    articles = [{"title": f"NEWS {i}", "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2025-05-08")} for i in range(5)]

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = articles

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=3)

    assert out.count("### ") == 3


@pytest.mark.unit
@pytest.mark.parametrize("field", ["title", "summary", "provider", "canonicalUrl", "pubDate"])
def test_an_article_with_a_null_field_is_read_with_its_default(field):
    """Yahoo 可能将某些字段设为 null，而非直接省略（#1458）。"""
    content = {"title": "T", "summary": "S", "provider": {"displayName": "P"},
               "canonicalUrl": {"url": "https://x"}, "pubDate": "2026-09-22T10:00:00Z", field: None}
    data = ynews._extract_article_data({"content": content})
    assert isinstance(data["title"], str) and isinstance(data["summary"], str)
    assert isinstance(data["publisher"], str) and isinstance(data["link"], str)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["title", "summary", "publisher", "link"])
def test_a_flat_article_with_a_null_field_is_read_with_its_default(field):
    data = ynews._extract_article_data({"title": "T", "summary": "S", "publisher": "P", "link": "L",
                                        "providerPublishTime": _epoch("2026-09-22"), field: None})
    assert all(isinstance(data[k], str) for k in ("title", "summary", "publisher", "link"))


@pytest.mark.unit
def test_one_malformed_article_does_not_lose_the_feed(monkeypatch):
    good = {"content": {"title": "Kept", "summary": "", "provider": {"displayName": "Wire"},
                        "pubDate": "2026-09-22T10:00:00Z"}}
    bad = {"content": {"title": "Also kept", "summary": None, "provider": None,
                       "pubDate": "2026-09-22T11:00:00Z"}}
    _ticker_with([good, bad], monkeypatch)
    out = ynews.get_news_yfinance("ZS", "2026-09-20", "2026-09-23")
    assert "Kept" in out and "Also kept" in out
