"""针对指定 ticker 的 Reddit 讨论帖子获取器。

通过 Reddit 公开 Atom/RSS 搜索流，一次请求检索多个社区。
匿名访问 JSON 搜索接口可能被 WAF 拒绝（HTTP 403，#862），
因此使用无需登录的 RSS 路径；该数据不包含投票或评论数。
触发 HTTP 429 时尊重 Retry-After 并退避重试一次。

请求失败必须标记为 ``<unavailable>``，不能误报“没有帖子”；
否则情绪分析师会把从未观测到的缺失当成真实信号（#1295）。
无需 API Key，返回可注入 Prompt 的纯文本；失败时返回占位字符串，
不让上游专门处理缺失或异常。"""

from __future__ import annotations

import html
import http.client
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)


def _within_window(posts, start_date, end_date):
    """仅保留 [start_date, end_date] 内发布的帖子，避免前视偏差。
    实时调用未提供时间窗口时不改变列表。
    历史窗口内缺少 ``created_utc`` 的帖子被剔除（#1220）。"""
    if not (start_date and end_date):
        return posts
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    return [p for p in posts if in_window(_posted_at(p), start_dt, end_dt)]


def _posted_at(post) -> datetime | None:
    """将帖子的 ``created_utc`` 时间戳转换为 UTC datetime；缺失返回 None。"""
    ts = post.get("created_utc")
    return datetime.fromtimestamp(ts, tz=UTC) if ts else None


def _coverage_dates(posts) -> list:
    """计算数据流的可观测日期范围。搜索固定在最近一周（``t=week``），
    因此即使返回空列表，也能把一周前作为默认覆盖起点。
    若结果达到整页上限，可能还有更早记录被截断，此时只能依赖
    实际返回的帖子时间确定覆盖范围。"""
    dates = [_posted_at(p) for p in posts]
    if len(posts) < _FEED_PAGE:
        dates.append(datetime.now(UTC) - _SEARCH_LOOKBACK)
    return dates


_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
# 按 Reddit 接口规范使用可识别客户端身份的 User-Agent。
# Reddit 可能拒绝 Mozilla/5.0 或 curl 等泛化标识；
# 该标识对公开 RSS 可用，无需伪装成浏览器。
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# 默认社区大致按指定 ticker 的信号密度排序。
# wallstreetbets 帖子量大但噪声较多，stocks、investing 相对稳健；
# 调用方可以覆盖此列表。
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")

# 加密资产主要在币圈社区讨论，尤其在币种专属社区；
# 一般股票社区很少提及这些币种。
CRYPTO_SUBREDDITS = ("CryptoCurrency", "CryptoMarkets")
_COIN_SUBREDDITS = {"BTC": "Bitcoin", "ETH": "ethereum", "SOL": "solana"}


def subreddits_for(ticker: str) -> tuple[str, ...]:
    """返回指定 ticker 需要检索的社区；加密资产使用相应币圈社区。"""
    base = crypto_base(ticker)
    if not base:
        return DEFAULT_SUBREDDITS
    coin = _COIN_SUBREDDITS.get(base)
    return ((coin,) if coin else ()) + CRYPTO_SUBREDDITS

# 采用 Reddit 单页最大条数。通常足够覆盖默认社区一周内的相关帖子，
# 避免高流量社区在组合搜索中挤占其他社区的结果。
_FEED_PAGE = 100
_SCREEN_CHARS = 1000   # of a post's title and body sent for screening


_SEARCH_LOOKBACK = timedelta(days=7)  # matches t=week below


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """将 Atom 的 ``published`` 时间解析为 UTC 时间戳，失败返回 None。"""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """将 Reddit Atom 条目中的 HTML 正文清洗为纯文本。"""
    if not content:
        return ""
    # Reddit 将帖子实际正文包在 SC_OFF / SC_ON 标记之间。
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


# 没有 Retry-After 头时的 429 退避策略。
# 测试发现间隔 8、10、30 秒重试仍可能得到 429，
# 等待约 60 秒才成功（#1295）。增加随机抖动，
# 避免同 IP 多个运行同时撞上限流窗口。
_RETRY_FALLBACK_SECONDS = 60.0


def _jitter(seconds: float, frac: float = 0.2) -> float:
    """对 ``seconds`` 增加 ±``frac`` 随机抖动，避免并发任务
    在相同 IP 限流窗口内同步重试。"""
    return seconds * (1.0 + random.uniform(-frac, frac))


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """解析 HTTP 429 响应中的 ``Retry-After`` 等待时间，最多 60 秒。

    上限与 ``_RETRY_FALLBACK_SECONDS`` 一致；
    如果服务端要求等待更久但客户端提前重试，只会浪费唯一一次重试。
    只有响应头缺失或无法解析才返回 None；
    有效的 ``Retry-After: 0`` 应返回 0.0，表示立即重试。"""
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 60.0) if val is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


# Reddit 搜索流通常只有一页数据，必须限制最大读取长度，
# 防止异常端点无限流式发送内容占满内存。
# 超限抛出 HTTPException，并按获取失败进行降级。
_MAX_FEED_BYTES = 5 * 1024 * 1024


def _read_capped(resp) -> bytes:
    """读取响应正文，大小不能超过 ``_MAX_FEED_BYTES``，超限时抛异常。"""
    data = resp.read(_MAX_FEED_BYTES + 1)
    if len(data) > _MAX_FEED_BYTES:
        raise http.client.HTTPException(
            f"Reddit feed exceeded {_MAX_FEED_BYTES} bytes; refusing to parse"
        )
    return data


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict] | None:
    """默认路径：解析 Reddit 公开 Atom 搜索流。

    ``sub`` 可为单个社区，或用 ``+`` 连接的多个社区。
    遇到每 IP 限流（429）时，按 ``Retry-After`` 或默认等待一次后重试。
    确实执行查询但没有匹配结果返回 []；网络获取失败返回 None。
    两者必须区分，否则会把请求失败当成“没有讨论”（#1295）。"""
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(_read_capped(resp))
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            # 严格遵守服务端 Retry-After（包括 0）；
            # 只有自定义默认等待时间才添加抖动，避免多个运行同步重试。
            retry_after = _retry_after_seconds(exc)
            wait = retry_after if retry_after is not None else _jitter(_RETRY_FALLBACK_SECONDS)
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError 涵盖 URLError、TimeoutError 及连接重置；
        # HTTPException 涵盖 IncompleteRead、BadStatusLine 等分块传输错误（#1024）。
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        category_el = entry.find("atom:category", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            # 组合搜索会为每条记录标明社区；单社区结果可能没有此字段，
            # 此时只能属于当前请求的社区。
            "subreddit": category_el.get("term") if category_el is not None
            else (sub if "+" not in sub else ""),
        })
    return posts


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] | None = None,
    *,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
    screen=None,
) -> str:
    """跨金融相关社区检索指定 ``ticker`` 的近期 Reddit 帖子，
    将其整理为可注入 Prompt 的纯文本区块。

    默认由 ``subreddits_for(ticker)`` 选择社区，并以
    ``r/a+b+c`` 组合请求一次查询，避免匿名 RSS 每 IP 约一分钟一次的
    频率限制；每条结果仍按其原社区分组。

    提供 start_date/end_date 时按历史窗口筛选，防止回测
    看到现在的讨论（#1220）。提供 ``screen`` 时，筛选器
    针对标题和正文返回每条帖子的保留标志，并附加说明行。
    筛选在每社区条数上限之前执行，使有效帖子能够填满展示位置。"""
    subreddits = list(subreddits or subreddits_for(ticker))
    # 系统内的加密资产代码是 Yahoo 交易对（如 BTC-USD），
    # Reddit 查询应使用基础币种 BTC 才能匹配正常讨论。
    ticker = crypto_base(ticker) or ticker
    label = ", ".join(f"r/{s}" for s in subreddits)
    fetched = _fetch_subreddit_rss(ticker, "+".join(subreddits), _FEED_PAGE, timeout)
    if fetched is None:
        return f"<Reddit unavailable: fetch failed ({label}); this is not an absence of discussion>"

    window = bool(start_date and end_date)
    posts = _within_window(fetched, start_date, end_date)
    if not posts:
        gap = window and coverage_gap(
            _coverage_dates(fetched), start_date, end_date,
            "Reddit search", f"discussion of {ticker.upper()}",
        )
        period = f"within {start_date}..{end_date}" if window else "in the past 7 days"
        return gap or f"<no Reddit posts found mentioning {ticker.upper()} across {label} {period}>"

    def sub_of(p):
        return p.get("subreddit") or (subreddits[0] if len(subreddits) == 1 else "unknown")

    note, screened_out = "", set()
    if screen:
        keep, note = screen([f"{p.get('title') or ''}\n{p.get('selftext') or ''}"[:_SCREEN_CHARS]
                             for p in posts])
        screened_out = {sub_of(p).lower() for p, kept in zip(posts, keep, strict=True) if not kept}
        posts = [p for p, kept in zip(posts, keep, strict=True) if kept]

    # 根据条目标记的社区按请求顺序分组，不能丢弃内容。
    # 单社区请求中未标记来源的帖子归入该社区，
    # 其他社区名称也单独保留对应区块。
    by_sub = {s.lower(): (s, []) for s in subreddits}
    for p in posts:
        by_sub.setdefault(sub_of(p).lower(), (sub_of(p), []))[1].append(p)

    page_full = len(fetched) >= _FEED_PAGE
    blocks = []
    for sub, sub_posts in by_sub.values():
        if not sub_posts:
            if sub.lower() in screened_out:
                blocks.append(f"r/{sub}: <no posts about {ticker.upper()} after screening>")
            else:
                blocks.append(
                    f"r/{sub}: <not among the newest {_FEED_PAGE} matches across {label}>"
                    if page_full else f"r/{sub}: <no posts found mentioning {ticker.upper()}>"
                )
            continue
        sub_posts = sub_posts[:limit_per_sub]  # the feed is newest-first
        lines = [f"r/{sub} — {len(sub_posts)} recent posts mentioning {ticker.upper()}:"]
        for p in sub_posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            created = p.get("created_utc")
            created_str = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{created_str}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(([note] if note else []) + blocks)
