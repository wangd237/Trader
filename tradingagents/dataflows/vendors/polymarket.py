"""Polymarket 预测市场数据提供商。

向新闻分析师提供未来事件（美联储决策、经济衰退、选举、地缘政治、
加密资产等）的实时市场隐含概率，补充新闻报道的已发生事件和 FRED
宏观数据反映的当前状态，提供市场对未来的定价预期。

调用公开的 Gamma API（https://gamma-api.polymarket.com），
无需密钥或登录。``outcomePrices`` 表示各结果的隐含概率，
例如 Yes 对应 0.76，表示市场定价约 76% 的发生概率。"""
import json
import logging
from datetime import UTC, datetime

import requests

from tradingagents.dataflows.date_window import is_historical

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# 网络超时秒数，与其他数据提供商保持一致。
REQUEST_TIMEOUT = 30

# 默认返回的市场数量，按照交易量排序。
DEFAULT_LIMIT = 6


def _request(path: str, params: dict) -> dict:
    response = requests.get(
        f"{GAMMA_BASE}/{path}", params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _parse_json_list(value) -> list:
    """Gamma 将 ``outcomes`` 和 ``outcomePrices`` 编码为 JSON 字符串数组。"""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _is_forward_looking(market: dict, now: datetime) -> bool:
    """仅保留尚未结束、将在未来结算的市场。

    ``closed`` 才是可靠的已结算标志；即使结算，``active`` 也可能为 True。
    ``endDate`` 早于今天也表示事件不再是前瞻信号，应排除。"""
    if market.get("closed"):
        return False
    end_date = market.get("endDate")
    if end_date:
        try:
            if datetime.fromisoformat(end_date.replace("Z", "+00:00")) < now:
                return False
        except ValueError:
            pass
    return bool(_parse_json_list(market.get("outcomePrices"))) and bool(
        _parse_json_list(market.get("outcomes"))
    )


def get_prediction_markets(topic: str, limit: int | None = None, as_of_date: str | None = None) -> str:
    """获取指定事件主题的实时预测市场隐含概率。

    参数：
        topic：主题关键词，例如 "Fed rate cut"、"recession 2026"、
            "US election"，也可为行业或公司事件。
        limit：按交易量排序的最大市场数；None 时使用 DEFAULT_LIMIT。
        as_of_date：分析日期。Polymarket 只提供实时赔率，历史日期不返回。

    返回：
        Markdown 报告，列出匹配主题、交易量最大的未结束市场，
        以及概率、成交量、结算日期和最近一周的变化。"""
    if is_historical(as_of_date):
        return (
            f"Prediction-market odds are withheld for {as_of_date}. Polymarket serves "
            f"only live odds on open markets, with no historical vintage, so serving "
            f"them would put post-decision information into a {as_of_date} analysis."
        )
    if limit is None:
        limit = DEFAULT_LIMIT

    try:
        data = _request("public-search", {"q": topic, "limit_per_type": 20})
    except requests.RequestException as e:
        logger.warning("Polymarket search failed for %r: %s", topic, e)
        return (
            f"Polymarket data is currently unavailable (network error: {e}). "
            f"Proceed without prediction-market signal for '{topic}'."
        )

    now = datetime.now(UTC)
    candidates = [
        m
        for event in data.get("events", [])
        for m in event.get("markets", [])
        if _is_forward_looking(m, now)
    ]
    candidates.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)

    header = (
        f'## Polymarket prediction markets: "{topic}"\n'
        f"Live, market-implied probabilities (higher traded volume = deeper, "
        f"more reliable). A probability is the crowd's priced odds of the event, "
        f"not a forecast you should take as certain.\n\n"
    )

    if not candidates:
        return header + (
            f"No open prediction markets matched '{topic}'. Polymarket coverage "
            f"is concentrated in macro, political, geopolitical, and crypto "
            f"events; a specific equity may have none."
        )

    lines = []
    for m in candidates[:limit]:
        prices = _parse_json_list(m.get("outcomePrices"))
        outcomes = _parse_json_list(m.get("outcomes"))
        try:
            prob = float(prices[0])
        except (ValueError, IndexError):
            continue
        label = outcomes[0] if outcomes else "Yes"
        volume = m.get("volumeNum") or 0
        end_date = (m.get("endDate") or "")[:10]
        wk = m.get("oneWeekPriceChange")
        wk_str = (
            f", 1-week {wk * 100:+.1f}pp"
            if isinstance(wk, (int, float)) and wk
            else ""
        )
        lines.append(
            f"- **{m.get('question')}** — {label} {prob:.0%} "
            f"(${volume:,.0f} volume, resolves {end_date}{wk_str})"
        )

    return header + "\n".join(lines) + "\n"
