"""历史情绪分析不能泄漏今天的数据到回测（#1220）。

StockTwits、Reddit 仅提供近期帖子，历史运行必须按日期窗口过滤，
未覆盖该窗口时应明确标记不可用，不能把今天的评论当作历史情绪。
各来源共用 date_window.in_window。"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tradingagents.dataflows.date_window import in_window
from tradingagents.dataflows.vendors import reddit, stocktwits


class _JsonResp:
    """模拟返回 JSON 正文的最小 urlopen() 上下文管理器。"""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


# --- 共用时间窗口辅助函数 ---------------------------------------------------

@pytest.mark.unit
def test_in_window_bounds_and_exclusive_upper():
    start = datetime(2026, 5, 1)
    end = datetime(2026, 5, 9)
    assert in_window(datetime(2026, 5, 5, tzinfo=UTC), start, end) is True
    assert in_window(datetime(2026, 5, 9, 23, 59, tzinfo=UTC), start, end) is True
    # 结束日次日零点被排除，避免未来信息泄漏。
    assert in_window(datetime(2026, 5, 10, 0, 0, tzinfo=UTC), start, end) is False
    # 应转换带时区时间，而不是截掉偏移：05-10T01:00+05:00 等于 05-09T20:00Z。
    assert in_window(datetime.fromisoformat("2026-05-10T01:00:00+05:00"), start, end) is True


@pytest.mark.unit
def test_in_window_undated_excluded_in_backtest_kept_live():
    old = datetime(2026, 5, 9)
    assert in_window(None, datetime(2026, 5, 1), old) is False       # historical
    now = datetime.now(UTC)
    assert in_window(None, now, now) is True                          # live


# --- StockTwits 数据源 -------------------------------------------------------------

def _msg(created_iso, sentiment=None):
    return {
        "created_at": created_iso,
        "user": {"username": "u"},
        "entities": {"sentiment": {"basic": sentiment}},
        "body": "text",
    }


@pytest.mark.unit
def test_stocktwits_historical_window_excludes_recent(monkeypatch):
    # 全部消息来自今天，不得展示给历史日期运行。
    recent = [_msg("2026-08-30T12:00:00Z", "Bullish"), _msg("2026-08-29T09:00:00Z")]
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: _JsonResp({"messages": recent}))
    out = stocktwits.fetch_stocktwits_messages("AAPL", start_date="2026-05-01", end_date="2026-05-08")
    assert "2026-05-01..2026-05-08" in out
    assert "Bullish: 1" not in out  # the recent bullish message did not leak
    # 数据覆盖起点晚于窗口，应报告不可用而非当时没有讨论。
    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_stocktwits_live_window_keeps_in_range(monkeypatch):
    msgs = [_msg("2026-05-05T12:00:00Z", "Bullish"), _msg("2026-05-07T09:00:00Z", "Bearish")]
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: _JsonResp({"messages": msgs}))
    out = stocktwits.fetch_stocktwits_messages("AAPL", start_date="2026-05-01", end_date="2026-05-08")
    assert "Total: 2" in out


@pytest.mark.unit
def test_stocktwits_no_window_is_unfiltered(monkeypatch):
    msgs = [_msg("2026-08-30T12:00:00Z", "Bullish")]
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: _JsonResp({"messages": msgs}))
    out = stocktwits.fetch_stocktwits_messages("AAPL")  # live caller, no dates
    assert "Total: 1" in out


# --- Reddit 数据源 -----------------------------------------------------------------

def _epoch(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())


@pytest.mark.unit
def test_reddit_historical_window_excludes_recent(monkeypatch):
    posts = [{"title": "NOW", "created_utc": _epoch("2026-08-30"), "source": "rss"}]
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: posts)
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",),
        start_date="2026-05-01", end_date="2026-05-08",
    )
    assert "NOW" not in out
    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_reddit_live_window_keeps_in_range(monkeypatch):
    posts = [{"title": "INRANGE", "created_utc": _epoch("2026-05-05"), "source": "rss"}]
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: posts)
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",),
        start_date="2026-05-01", end_date="2026-05-08",
    )
    assert "INRANGE" in out


# --- 覆盖范围与真正无消息的区别 --------------------------------------------------------
# The public feeds only serve recent items. When everything fetched postdates the
# window the source cannot answer for that date; reporting "no posts" there is a
# claim about the market that was never observed.

@pytest.mark.unit
def test_stocktwits_covered_but_empty_window_is_a_real_absence(monkeypatch):
    # 数据流覆盖到窗口之前但窗口内为空，才可能认定真实没有帖子。
    msgs = [_msg("2026-08-30T12:00:00Z"), _msg("2026-04-20T12:00:00Z")]
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: _JsonResp({"messages": msgs}))
    out = stocktwits.fetch_stocktwits_messages("AAPL", start_date="2026-05-01", end_date="2026-05-08")
    assert "no StockTwits messages" in out
    assert "unavailable" not in out


@pytest.mark.unit
def test_reddit_covered_but_empty_window_is_a_real_absence(monkeypatch):
    posts = [{"title": "NOW", "created_utc": _epoch("2026-08-30"), "source": "rss"},
             {"title": "OLD", "created_utc": _epoch("2026-04-20"), "source": "rss"}]
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: posts)
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",),
        start_date="2026-05-01", end_date="2026-05-08",
    )
    assert "no reddit posts" in out.lower()
    assert "unavailable" not in out


@pytest.mark.unit
def test_reddit_empty_feed_for_an_old_window_is_unavailable(monkeypatch):
    # 搜索只覆盖最近一周，空结果不能证明数月前没有帖子。
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: [])
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",),
        start_date="2024-05-01", end_date="2024-05-08",
    )
    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_reddit_live_empty_feed_is_a_real_absence(monkeypatch):
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: [])
    out = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks",))
    assert "no reddit posts" in out.lower() and "past 7 days" in out
    assert "unavailable" not in out


@pytest.mark.unit
def test_stocktwits_empty_stream_for_a_past_window_is_unavailable(monkeypatch):
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: _JsonResp({"messages": []}))
    out = stocktwits.fetch_stocktwits_messages("AAPL", start_date="2026-05-01", end_date="2026-05-08")
    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_reddit_window_straddling_the_lookback_is_unavailable(monkeypatch):
    # 请求窗口横跨十天前至五天前，一周搜索无法覆盖前三天；
    # 空响应不足以证明整个窗口没有帖子。
    from datetime import timedelta
    today = datetime.now(UTC).date()
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: [])
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",),
        start_date=str(today - timedelta(days=10)), end_date=str(today - timedelta(days=5)),
    )
    assert "unavailable" in out


@pytest.mark.unit
def test_reddit_standard_week_window_empty_is_a_real_absence(monkeypatch):
    # Graph 窗口为 [trade_date - 7, trade_date]；
    # 当一周搜索覆盖全部窗口时，空响应才可表示没有帖子。
    from datetime import timedelta
    today = datetime.now(UTC).date()
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: [])
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",),
        start_date=str(today - timedelta(days=7)), end_date=str(today),
    )
    assert "no reddit posts" in out.lower() and "unavailable" not in out


@pytest.mark.unit
def test_reddit_full_page_does_not_vouch_for_older_days(monkeypatch):
    # 今天的 100 条帖子不能说明五天前是否有讨论；
    # 热门数据可能截断旧帖子，窗口仍应标记不可用。
    from datetime import timedelta
    today = datetime.now(UTC).date()
    ts = _epoch(str(today))
    page = [{"title": f"T{i}", "created_utc": ts, "subreddit": "stocks"} for i in range(reddit._FEED_PAGE)]
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a, **k: page)
    out = reddit.fetch_reddit_posts(
        "AAPL", subreddits=("stocks",),
        start_date=str(today - timedelta(days=6)), end_date=str(today - timedelta(days=5)),
    )
    assert "unavailable" in out and "no reddit posts" not in out.lower()
