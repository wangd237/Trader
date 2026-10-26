"""StockTwits 公开的交易标的消息流获取器。

StockTwits 为每个标的提供无需 API Key、OAuth 或注册的公开接口
``api.stocktwits.com/api/2/streams/symbol/{ticker}.json``。
每条消息包含用户标记的情绪（Bullish/Bearish/null）、
正文、时间戳和发布者。

本模块刻意保持独立：使用较短超时，HTTP 或解析失败时平稳降级，
并始终返回字符串，让下游 Agent 无需区分网络是否成功。"""

from __future__ import annotations

import contextlib
import html
import http.client
import json
import logging
from datetime import datetime
from urllib.request import Request, urlopen

from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)

_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"


def _created_at(message) -> datetime | None:
    """解析消息的 ISO 8601 ``created_at`` 时间；缺失或错误时返回 None。"""
    raw = message.get("created_at")
    if not raw:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return None


def _within_window(messages, start_date, end_date):
    """只保留位于 [start_date, end_date] 的帖子，避免未来信息泄漏。

    开始和结束日期均为空时，实时调用保留全部消息。
    在历史窗口内，无法解析 ``created_at`` 的帖子必须丢弃，
    因为无法确认它是否在分析日之后发布（#1220）。"""
    if not (start_date and end_date):
        return messages
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    return [m for m in messages if in_window(_created_at(m), start_dt, end_dt)]


def _body(message) -> str:
    """解析消息正文；接口返回的是 HTML 转义文本（如 ``&amp;``、``&#39;``）。"""
    return html.unescape(message.get("body") or "")


def _stocktwits_symbol(ticker: str) -> str:
    """将加密资产交易对转换为 StockTwits 使用的 ``<BASE>.X`` 代码。

    StockTwits 使用 ``BTC.X``，而 Yahoo 的 ``BTC-USD`` 格式会返回 404。
    因此加密资产统一转换为基础币种加 ``.X``，
    其他标的只转换为大写。"""
    base = crypto_base(ticker)
    return f"{base}.X" if base else ticker.strip().upper()


def fetch_stocktwits_messages(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
    screen=None,
) -> str:
    """获取指定 ``ticker`` 的近期 StockTwits 消息，
    生成可直接注入 Prompt 的纯文本区块。

    提供 ``start_date`` / ``end_date`` 时先按历史日期窗口筛选，
    避免回测看到今天的讨论（#1220）。公开消息流只包含近期帖子，
    超出覆盖范围的窗口应标记为“不可用”，而不能误报“没有帖子”。

    提供 ``screen`` 时，筛选器接收每条正文并返回保留标志，
    同时生成置于区块顶部的说明行。

    接口不可达、代码没有帖子或响应结构异常时，
    统一返回占位字符串，调用者不必处理 None 或异常。"""
    url = _API.format(ticker=_stocktwits_symbol(ticker))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        # OSError 涵盖 URLError、TimeoutError 和连接重置；
        # HTTPException 则用于处理分块传输错误
        # （IncompleteRead/BadStatusLine，#1024）。
        logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
        return f"<stocktwits unavailable: {type(exc).__name__}>"

    fetched = data.get("messages", []) if isinstance(data, dict) else []
    messages = _within_window(fetched, start_date, end_date)
    if not messages:
        if start_date and end_date:
            gap = coverage_gap(
                (_created_at(m) for m in fetched), start_date, end_date,
                "StockTwits", f"messages about ${ticker.upper()}",
            )
            return gap or (
                f"<no StockTwits messages for ${ticker.upper()} within "
                f"{start_date}..{end_date}>"
            )
        return f"<no StockTwits messages found for ${ticker.upper()}>"

    note = ""
    if screen:
        keep, note = screen([_body(m) for m in messages])
        screened = len(messages)
        messages = [m for m, kept in zip(messages, keep, strict=True) if kept]
        if not messages:
            return f"{note}\n\n<none of the {screened} StockTwits messages is about ${ticker.upper()}>"

    lines = []
    bullish = bearish = unlabeled = 0
    for m in messages[:limit]:
        created = m.get("created_at", "")
        user = (m.get("user") or {}).get("username", "?")
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        body = _body(m).replace("\n", " ").strip()
        if len(body) > 280:
            body = body[:280] + "…"

        if sentiment == "Bullish":
            bullish += 1
            tag = "Bullish"
        elif sentiment == "Bearish":
            bearish += 1
            tag = "Bearish"
        else:
            unlabeled += 1
            tag = "no-label"
        lines.append(f"[{created} · @{user} · {tag}] {body}")

    total = bullish + bearish + unlabeled
    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0
    summary = (
        f"Bullish: {bullish} ({bull_pct}%) · "
        f"Bearish: {bearish} ({bear_pct}%) · "
        f"Unlabeled: {unlabeled} · "
        f"Total: {total} most-recent messages"
    )
    return (f"{note}\n\n" if note else "") + summary + "\n\n" + "\n".join(lines)
