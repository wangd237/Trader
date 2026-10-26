"""基于 yfinance 的新闻数据获取函数。"""

import contextlib
from datetime import UTC, datetime

import yfinance as yf
from dateutil.relativedelta import relativedelta

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry


def _extract_article_data(article: dict) -> dict:
    """从 yfinance 的新闻结构中提取文章数据，兼容嵌套的 content 格式。

    Yahoo 可能为部分字段返回 null 而不是直接省略，
    因此无论字段缺失还是值为 null，都应使用默认值（#1458）。"""
    content = article.get("content")
    if isinstance(content, dict):
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        pub_date_str = content.get("pubDate") or ""
        pub_date = None
        if pub_date_str:
            with contextlib.suppress(ValueError, AttributeError):
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))

        return {
            "title": content.get("title") or "No title",
            "summary": content.get("summary") or "",
            "publisher": (content.get("provider") or {}).get("displayName") or "Unknown",
            "link": url_obj.get("url") or "",
            "pub_date": pub_date,
        }
    # 兼容扁平结构的新闻记录，并解析其 Unix 发布时间，
    # 保证历史窗口筛选同样生效，防止未来新闻泄漏（#992/#1007）。
    pub_date = None
    ts = article.get("providerPublishTime")
    if ts:
        # Unix 秒数采用 UTC；解析时保留 UTC 时区信息，
        # 避免不同服务器本地时区导致筛选结果偏移（#1126）。
        with contextlib.suppress(ValueError, OSError, TypeError):
            pub_date = datetime.fromtimestamp(ts, tz=UTC)
    return {
        "title": article.get("title") or "No title",
        "summary": article.get("summary") or "",
        "publisher": article.get("publisher") or "Unknown",
        "link": article.get("link") or "",
        "pub_date": pub_date,
    }


def _search_news(canonical: str, limit: int) -> tuple[list, bool]:
    """提取 Yahoo Search 中明确标记为 ``canonical`` 的文章，
    并判断服务是否认识该标的。

    即使报价消息流为空，搜索也可能返回文章；但相关性搜索结果
    只是样本，且标签可能较宽泛，只有明确关联代码的文章才能视为该标的新闻。"""
    search = yf_retry(lambda: yf.Search(canonical, news_count=limit))
    if search is None:  # Yahoo answered "not found": its search is down
        return [], False
    tagged = [
        a for a in (search.news or [])
        if canonical in (t.upper() for t in a.get("relatedTickers") or [])
    ]
    knows = any((q.get("symbol") or "").upper() == canonical for q in search.quotes or [])
    return tagged, knows or bool(search.news)


def _format_articles(articles: list[dict]) -> str:
    out = ""
    for data in articles:
        out += f"### {data['title']} (source: {data['publisher']})\n"
        if data["summary"]:
            out += f"{data['summary']}\n"
        if data["link"]:
            out += f"Link: {data['link']}\n"
        out += "\n"
    return out


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """获取 Yahoo Finance 中分析窗口内指定 ``ticker`` 的新闻。

    优先使用该标的报价消息流；为空时从 Yahoo Search 中筛选
    明确标记该代码的文章。若找不到或均无响应，代表 Yahoo 新闻源不可用，
    不能据此判断市场没有新闻，应尝试下一个已配置提供商。
    搜索只覆盖部分样本，窗口内搜索为空也不能直接推断没有新闻。"""
    article_limit = get_config()["news_article_limit"]
    # 与其他 yfinance 数据接口一样，使用标准化 ticker 查询 Yahoo；
    # 原始券商、外汇或加密资产别名（如 XAUUSD、BTCUSD）可能静默查不到新闻。
    # 报告标题仍保留用户原始输入代码。
    canonical = normalize_symbol(ticker)
    resolved = "" if canonical == ticker else f" (resolved to {canonical})"
    subject = f"news for {ticker}{resolved}"
    feed = yf_retry(lambda: yf.Ticker(canonical).get_news(count=article_limit)) or []
    articles = [_extract_article_data(a) for a in feed]
    if not feed:
        found, answered = _search_news(canonical, article_limit)
        if not answered:
            raise VendorUnavailableError(f"Yahoo Finance returned no news response for {canonical}")
        if not found:
            # 当报价消息流为空时，搜索不到带该代码标签的文章
            # 不能说明公司确实没有新闻；Yahoo 对大量非美股缺少文章标签。
            raise VendorUnavailableError(f"Yahoo Finance has no news articles tagged {canonical}")
        articles = [_extract_article_data(a) for a in found]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    # 仅保留位于请求日期窗口内的文章，防止未来信息泄漏。
    in_range = [a for a in articles if in_window(a["pub_date"], start_dt, end_dt)]
    if in_range:
        return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{_format_articles(in_range)}"

    if not feed:
        return (f"<Yahoo Finance news unavailable for {start_date}..{end_date}: Yahoo search "
                f"returns only a sample of recent articles about {canonical}, so this is not "
                f"an absence of {subject}>")
    gap = coverage_gap(
        (a["pub_date"] for a in articles),
        start_date, end_date, "Yahoo Finance news", subject,
    )
    return gap or f"No news found for {ticker}{resolved} between {start_date} and {end_date}"


def get_global_news_yfinance(
    as_of_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """通过 yfinance Search 获取全球及宏观经济新闻。

    参数：
        as_of_date：分析日期，格式 yyyy-mm-dd。
        look_back_days：回溯天数；None 使用当前配置的
            ``global_news_lookback_days``。
        limit：最大文章数；None 使用当前配置的
            ``global_news_article_limit``。

    返回：
        包含全球新闻文章的格式化字符串。"""
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    search_queries = config["global_news_queries"]

    curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    in_window_news = []
    seen_titles = set()

    for query in search_queries:
        found = yf_retry(lambda q=query: yf.Search(
            query=q,
            news_count=limit,
            enable_fuzzy_query=True,
        ).news)

        for article in found or []:
            # 先按日期窗口筛选，再应用数量限制。
            # 窗口外的文章不能占用条数预算或提前停止后续搜索（#1356）。
            # 扁平格式的文章也必须遵守同一规则，避免未来新闻泄漏（#1007）。
            data = _extract_article_data(article)
            if not in_window(data["pub_date"], start_dt, curr_dt):
                continue
            if data["title"] and data["title"] not in seen_titles:
                seen_titles.add(data["title"])
                in_window_news.append(data)

        if len(in_window_news) >= limit:
            break

    news_str = _format_articles(in_window_news[:limit])

    # 窗口内没有文章时，应明确说明，而不是返回只有标题的空报告（#993）。
    if not news_str:
        # 结果合并了多组模糊搜索，时间戳不能证明持续的历史覆盖；
        # 窗口覆盖情况只能根据当前可见数据作保守判断。
        gap = coverage_gap((), start_date, as_of_date, "Yahoo Finance global news", "market news")
        return gap or f"No global news found between {start_date} and {as_of_date}"

    return f"## Global Market News, from {start_date} to {as_of_date}:\n\n{news_str}"
