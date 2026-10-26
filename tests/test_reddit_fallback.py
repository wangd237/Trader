"""Reddit RSS 获取器回归测试：组合请求、429 退避重试，
以及分块传输异常降级（#1024）。"""

from __future__ import annotations

import http.client
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows.vendors import reddit

_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>NVDA earnings beat, stock pops</title>
    <published>2026-05-20T14:30:00+00:00</published>
    <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Great &lt;b&gt;quarter&lt;/b&gt; for NVDA&amp;#39;s datacenter unit.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  </entry>
  <entry>
    <title>Is NVDA overvalued?</title>
    <published>2026-05-19T09:00:00Z</published>
    <content type="html">&lt;p&gt;Forward P/E discussion&lt;/p&gt;</content>
  </entry>
</feed>
"""


def _resp(read_fn):
    """最小化上下文管理器响应，其 read() 调用 ``read_fn``。"""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner, size=-1):
            data = read_fn()
            return data if size is None or size < 0 else data[:size]
    return _Resp()


def _atom_resp():
    return _resp(lambda: _SAMPLE_ATOM.encode("utf-8"))


def _raise(exc):
    def _r():
        raise exc
    return _resp(_r)


@pytest.mark.unit
class TestIsoToTimestamp:
    def test_parses_offset_and_z(self):
        assert reddit._iso_to_timestamp("2026-05-20T14:30:00+00:00") > 0
        assert reddit._iso_to_timestamp("2026-05-19T09:00:00Z") > 0

    def test_none_and_garbage_return_none(self):
        assert reddit._iso_to_timestamp(None) is None
        assert reddit._iso_to_timestamp("not-a-date") is None


@pytest.mark.unit
class TestStripHtml:
    def test_extracts_between_sc_markers_and_unescapes(self):
        raw = "<!-- SC_OFF --><div class=\"md\"><p>Great <b>quarter</b> &amp; more</p></div><!-- SC_ON -->"
        assert reddit._strip_html(raw) == "Great quarter & more"

    def test_empty(self):
        assert reddit._strip_html("") == ""


@pytest.mark.unit
class TestRssParsing:
    def test_parses_atom_entries(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", limit=5, timeout=5.0)
        assert len(posts) == 2
        assert posts[0]["title"] == "NVDA earnings beat, stock pops"
        assert posts[0]["created_utc"] > 0
        assert "datacenter unit" in posts[0]["selftext"]
        assert posts[0]["subreddit"] == "stocks"

    def test_malformed_xml_reports_unavailable(self):
        with patch.object(reddit, "urlopen", return_value=_resp(lambda: b"<<not xml>>")):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None


@pytest.mark.unit
class TestRss429Backoff:
    def test_429_then_success_retries_once(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]) as op, \
             patch.object(reddit.time, "sleep") as slept:
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # original + exactly one retry
        slept.assert_called_once()         # backed off before retrying
        assert len(posts) == 2

    def test_429_twice_gives_up_after_one_retry(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, err]) as op, \
             patch.object(reddit.time, "sleep"):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # one retry, then gives up cleanly
        assert posts is None

    def test_retry_after_header_is_honoured(self):
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "12"}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once_with(12.0)

    def test_retry_after_zero_is_honoured_not_treated_as_absent(self):
        # 合法的 Retry-After: 0 表示立即重试，
        # 不能因使用 `or` 将 0 错误转成默认等待。
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "0"}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once_with(0.0)

    def test_headerless_429_fallback_is_jittered(self):
        # 没有 Retry-After 时采用带随机抖动的默认退避，
        # 避免并发运行同时重试。
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once()
        (wait,), _ = slept.call_args
        assert 48.0 <= wait <= 72.0  # 60s +/-20% jitter


@pytest.mark.unit
class TestChunkedTransferErrorsHandled:
    """IncompleteRead 和 RemoteDisconnected 来自 http.client，
    不是 OSError；此前异常未被捕获，可能使整个流程崩溃（#1024）。"""

    def test_rss_incomplete_read_reports_unavailable(self):
        with patch.object(reddit, "urlopen", return_value=_raise(http.client.IncompleteRead(b""))):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None

    def test_oversized_rss_feed_is_refused_not_parsed(self):
        # 恶意或异常端点可能无限流式发送正文，
        # 必须限制读取大小，超限则按获取失败降级。
        big = _resp(lambda: b"x" * 100)
        with patch.object(reddit, "_MAX_FEED_BYTES", 10), \
             patch.object(reddit, "urlopen", return_value=big):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None


@pytest.mark.unit
class TestFormatterHandlesRssPosts:
    def test_rss_posts_omit_fake_counts_and_note_source(self):
        rss_posts = [{
            "title": "NVDA pops", "score": None, "num_comments": None,
            "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
            "selftext": "great quarter", "source": "rss",
        }]
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=rss_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert "↑" not in out  # RSS has no scores; none are invented
        assert "NVDA pops" in out
        assert "great quarter" in out


@pytest.mark.unit
class TestCryptoSearchTerm:
    """Reddit 中很少直接使用 BTC-USD 交易对形式，应搜索基础币种（#1113）。"""

    def _captured_ticker(self, ticker):
        seen = {}

        def fake_fetch(t, subs, limit, timeout, **kwargs):
            seen["ticker"] = t
            return []

        with patch.object(reddit, "_fetch_subreddit_rss", side_effect=fake_fetch):
            reddit.fetch_reddit_posts(ticker, subreddits=("stocks",))
        return seen["ticker"]

    def test_crypto_pair_searches_base(self):
        assert self._captured_ticker("BTC-USD") == "BTC"

    def test_equity_passes_through(self):
        assert self._captured_ticker("NVDA") == "NVDA"


@pytest.mark.unit
class TestOneRequestForAllSubreddits:
    """Reddit 匿名 RSS 每 IP 每分钟大约允许一个请求。
    逐社区请求会频繁进入退避；使用 r/a+b+c 的组合流一次检索，
    通过条目中的社区来源区分结果，避免丢失社区信息。"""

    def _post(self, sub, title="NVDA pops"):
        return {"title": title, "score": None, "num_comments": None,
                "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
                "selftext": "", "source": "rss", "subreddit": sub}

    def test_all_subreddits_share_one_request(self):
        calls = []

        def record(t, subs, limit, timeout):
            calls.append((subs, limit))
            return []

        with patch.object(reddit, "_fetch_subreddit_rss", side_effect=record):
            reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b", "c"), limit_per_sub=5)
        # 组合搜索使用完整一页，降低热门社区挤占其他社区结果的风险。
        assert calls == [("a+b+c", reddit._FEED_PAGE)]

    def test_posts_are_grouped_back_by_subreddit(self):
        posts = [self._post("b", "FROM B"), self._post("a", "FROM A")]
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"))
        assert out.index("r/a") < out.index("FROM A") < out.index("r/b") < out.index("FROM B")

    def test_failed_request_is_unavailable_not_silence(self):
        # #1295：限流失败不能解释为确实没有帖子。
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=None):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"))
        assert "Reddit unavailable" in out
        assert "no Reddit posts found" not in out

    def test_genuine_empty_still_reports_no_posts(self):
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=[]):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"))
        assert "no Reddit posts found" in out
        assert "unavailable" not in out

    def test_subreddit_with_no_posts_is_listed_when_others_have_some(self):
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=[self._post("a")]):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"))
        assert "r/b: <no posts found" in out


@pytest.mark.unit
def test_posts_from_an_unrequested_or_unnamed_subreddit_are_not_dropped():
    posts = [
        {"title": "ELSEWHERE", "created_utc": None, "selftext": "", "subreddit": "options"},
        {"title": "NO LABEL", "created_utc": None, "selftext": "", "subreddit": ""},
    ]
    with patch.object(reddit, "_fetch_subreddit_rss", return_value=posts):
        out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"))
    assert "ELSEWHERE" in out and "r/options" in out
    assert "NO LABEL" in out


@pytest.mark.unit
def test_each_subreddit_keeps_its_own_quota():
    busy = [{"title": f"A{i}", "created_utc": None, "selftext": "", "subreddit": "a"} for i in range(9)]
    quiet = [{"title": "B0", "created_utc": None, "selftext": "", "subreddit": "b"}]
    with patch.object(reddit, "_fetch_subreddit_rss", return_value=busy + quiet):
        out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"), limit_per_sub=3)
    assert "A0" in out and "A2" in out and "A3" not in out  # capped per subreddit
    assert "B0" in out                                        # not crowded out


@pytest.mark.unit
def test_empty_subreddit_on_a_full_page_is_not_called_empty():
    # 满页时冷门社区帖子可能被截断，未返回结果不能证明没有讨论。
    full = [{"title": f"A{i}", "created_utc": None, "selftext": "", "subreddit": "a"}
            for i in range(reddit._FEED_PAGE)]
    with patch.object(reddit, "_fetch_subreddit_rss", return_value=full):
        out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"))
    assert "r/b: <no posts found" not in out
    assert f"newest {reddit._FEED_PAGE}" in out


def _screen_out(*dropped):
    """按帖子文本前缀筛选，丢弃以 ``dropped`` 中任意词开头的帖子。"""
    def screen(texts):
        return [not t.startswith(dropped) for t in texts], "Screened: note"
    return screen


@pytest.mark.unit
def test_screened_out_posts_free_their_subreddit_slots():
    posts = [{"title": t, "created_utc": None, "selftext": "", "subreddit": "a"}
             for t in ("SPAM1", "SPAM2", "A1", "A2")]
    with patch.object(reddit, "_fetch_subreddit_rss", return_value=posts):
        out = reddit.fetch_reddit_posts("NVDA", subreddits=("a",), limit_per_sub=2,
                                        screen=_screen_out("SPAM"))
    assert out.startswith("Screened: note")
    assert "A1" in out and "A2" in out and "SPAM" not in out


@pytest.mark.unit
def test_a_subreddit_emptied_by_screening_is_not_called_empty():
    posts = [{"title": "SPAM", "created_utc": None, "selftext": "", "subreddit": "b"},
             {"title": "A1", "created_utc": None, "selftext": "", "subreddit": "a"}]
    with patch.object(reddit, "_fetch_subreddit_rss", return_value=posts):
        out = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"), screen=_screen_out("SPAM"))
    assert "r/b: <no posts about NVDA after screening>" in out


@pytest.mark.unit
def test_an_unavailable_screen_keeps_every_post_and_says_so():
    posts = [{"title": "A1", "created_utc": None, "selftext": "", "subreddit": "a"}]

    def unavailable(texts):
        return [True] * len(texts), "<Jev screening unavailable (HTTP 529); posts are unscreened>"

    with patch.object(reddit, "_fetch_subreddit_rss", return_value=posts):
        screened = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"), screen=unavailable)
        plain = reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b"))
    assert screened == "<Jev screening unavailable (HTTP 529); posts are unscreened>\n\n" + plain


@pytest.mark.unit
@pytest.mark.parametrize("ticker, expected", [
    ("NVDA", ("wallstreetbets", "stocks", "investing")),
    ("BTC-USD", ("Bitcoin", "CryptoCurrency", "CryptoMarkets")),
    ("ETH-USD", ("ethereum", "CryptoCurrency", "CryptoMarkets")),
    ("SOL-USD", ("solana", "CryptoCurrency", "CryptoMarkets")),
    ("DOGE-USD", ("CryptoCurrency", "CryptoMarkets")),
])
def test_a_crypto_ticker_is_searched_in_crypto_communities(ticker, expected):
    assert reddit.subreddits_for(ticker) == expected


@pytest.mark.unit
def test_a_crypto_pair_is_fetched_from_its_communities_in_one_request():
    with patch.object(reddit, "_fetch_subreddit_rss", return_value=[]) as fetch:
        out = reddit.fetch_reddit_posts("ETH-USD")
    fetch.assert_called_once()
    query, subs = fetch.call_args.args[:2]
    assert (query, subs) == ("ETH", "ethereum+CryptoCurrency+CryptoMarkets")
    assert "r/wallstreetbets" not in out
