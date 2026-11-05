"""Chinese financial-news feeds as an optional first choice for global/US/HK news.

Tushare's Yicai news requires a separately authorised TUSHARE_TOKEN. Free
AKShare feeds from CLS, Eastmoney, and Sina are best-effort *recent* news, not
historical archives. Every returned item must have a usable publication date;
missing/old-window coverage falls back to the next configured vendor.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta

import pandas as pd

from tradingagents.dataflows.errors import VendorUnavailableError

logger = logging.getLogger(__name__)

# Conservative aliases: never infer another company's identity from its ticker.
_TICKER_ALIASES = {
    "BABA": ("阿里巴巴", "阿里集团", "Alibaba", "BABA", "淘宝", "淘天"),
    "9988.HK": ("阿里巴巴", "阿里集团", "Alibaba", "BABA", "淘宝", "淘天"),
}
_FEEDS = (
    ("财联社", "stock_info_global_cls"),
    ("东方财富", "stock_info_global_em"),
    ("新浪财经", "stock_info_global_sina"),
)


def _cell(row, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and pd.notna(value):
            cleaned = str(value).strip()
            if cleaned and cleaned.lower() not in ("nan", "nat", "none"):
                return cleaned
    return ""


def _timestamp(row) -> datetime | None:
    published = _cell(row, "datetime", "发布时间", "发布时间戳", "pub_date", "日期", "发布日期")
    date_only = _cell(row, "发布日期", "日期")
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", published) and date_only:
        published = f"{date_only} {published}"
    if not published:
        return None
    dt = pd.to_datetime(published, errors="coerce")
    if pd.isna(dt):
        return None
    try:
        if dt.tzinfo is not None:
            dt = dt.tz_convert("Asia/Shanghai").tz_localize(None)
        return dt.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _normalise(df: pd.DataFrame, source: str, start: datetime, end: datetime) -> list[dict]:
    if df is None or df.empty:
        return []
    results = []
    for row in df.to_dict(orient="records"):
        published = _timestamp(row)
        if published is None or not (start <= published <= end):
            continue
        title = _cell(row, "标题", "title")
        summary = _cell(row, "内容", "摘要", "content", "新闻内容")
        if not (title or summary):
            continue
        results.append({
            "time": published,
            "title": title or summary[:90],
            "summary": summary[:1000],
            "source": source,
            "link": _cell(row, "链接", "url", "URL", "新闻链接"),
        })
    return results


def _feed_items(start: datetime, end: datetime) -> list[dict]:
    items = []
    # First Financial / Yicai is available from Tushare only for accounts
    # that have purchased the news entitlement. Never silently claim access.
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        try:
            import tushare as ts

            client = ts.pro_api(token)
            data = client.news(
                src="yicai",
                start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            items.extend(_normalise(data, "第一财经 (Tushare)", start, end))
        except Exception as exc:
            logger.warning("Yicai/Tushare news unavailable: %s", type(exc).__name__)

    try:
        import akshare as ak
    except ImportError as exc:
        if items:
            return items
        raise VendorUnavailableError(
            "Chinese news requires AKShare (pip install -e '.[ashare]') "
            "or a Tushare news-entitled account"
        ) from exc

    for source, method in _FEEDS:
        try:
            data = getattr(ak, method)()
            items.extend(_normalise(data, source, start, end))
        except Exception as exc:
            # A single newspaper feed should not discard other providers.
            logger.warning("%s news unavailable: %s", source, type(exc).__name__)
    return items


def _unique(items: list[dict], limit: int) -> list[dict]:
    seen = set()
    output = []
    for item in sorted(items, key=lambda x: x["time"], reverse=True):
        key = (item["title"].strip().casefold(), item["time"].date())
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _render(items: list[dict], heading: str) -> str:
    lines = [heading]
    for item in items:
        lines.append(
            f"### {item['title']}\n"
            f"Published: {item['time']:%Y-%m-%d %H:%M:%S} (Asia/Shanghai)\n"
            f"Source: {item['source']}"
        )
        if item["summary"] and item["summary"] != item["title"]:
            lines.append(item["summary"])
        if item["link"]:
            lines.append(f"Link: {item['link']}")
    return "\n\n".join(lines)


def _range(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(microseconds=1)
    if start > end:
        raise ValueError("start_date is after end_date")
    return start, end


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Get dated Chinese-language articles concerning a ticker, if covered."""
    aliases = _TICKER_ALIASES.get(ticker.upper(), (ticker.upper(),))
    start, end = _range(start_date, end_date)
    items = _feed_items(start, end)
    pattern = re.compile("|".join(re.escape(x) for x in aliases), re.IGNORECASE)
    matches = [
        item for item in items
        if pattern.search(item["title"] + " " + item["summary"])
    ]
    matches = _unique(matches, 20)
    if not matches:
        raise VendorUnavailableError(
            f"Chinese news feeds have no verified {ticker} coverage for {start_date}..{end_date}"
        )
    return _render(matches, f"## {ticker} Chinese financial news ({start_date}..{end_date})")


def get_global_news(
    as_of_date: str, look_back_days: int | None = None, limit: int | None = None
) -> str:
    from tradingagents.dataflows.config import get_config

    cfg = get_config()
    days = cfg["global_news_lookback_days"] if look_back_days is None else look_back_days
    count = cfg["global_news_article_limit"] if limit is None else limit
    end_date = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=days)
    start, end = _range(start_date.strftime("%Y-%m-%d"), as_of_date)
    items = _unique(_feed_items(start, end), count)
    if not items:
        raise VendorUnavailableError(
            f"Chinese news feeds have no verifiably dated news for {start.date()}..{as_of_date}"
        )
    return _render(items, f"## Chinese global financial news ({start.date()}..{as_of_date})")
