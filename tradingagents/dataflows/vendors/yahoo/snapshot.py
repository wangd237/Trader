"""确定性的行情数据核验快照。

市场分析师是 LLM，可能编造精确指标数字，例如没有数据支持的布林带
或“历史验证的反弹”（#830）。本模块不调用 LLM，而是计算分析日期之前
最新 OHLCV、常见技术指标及近期收盘价，作为精确数字主张的事实依据。"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

# 使用固定的通用指标集合，使每次运行快照结构一致。
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _verified_rows(symbol: str, as_of_date: str) -> pd.DataFrame:
    """取得不晚于 as_of_date 的 OHLCV 并按日期排序；无可用数据时抛出 NoMarketDataError。

    虽然 ``load_ohlcv`` 已规范化日期并过滤未来数据，这里仍再次应用
    日期上限。作为数据核验路径，不能完全信任输入已预先过滤。"""
    # 直接使用原始报价：Agent 会将快照数字作为精确价格引用，
    # 填补缺失值后的数据可能错误地把上一交易日价格放在当前日期。
    data = load_ohlcv(symbol, as_of_date, fill_gaps=False)
    if data is None or data.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), "no price rows")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(as_of_date)].sort_values("Date")
    if df.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), f"no price rows on or before {as_of_date}")
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    as_of_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
    data: pd.DataFrame | None = None,
) -> str:
    """渲染真实行情快照，包括最新 OHLCV、技术指标及近期收盘价。"""
    # `df` 保留原始 OHLCV 大写列名（Open/High/Low/Close/Volume）；
    # stockstats 的 `wrap()` 会转为小写并加入指标列。
    # 因此原始价格读取 `df`，技术指标读取 `stock_df`。
    df = _verified_rows(symbol, as_of_date) if data is None else data.copy()
    if data is not None:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].le(pd.Timestamp(as_of_date))].sort_values('Date')
        if df.empty:
            raise NoMarketDataError(symbol, symbol, 'no A-share price rows as of date')
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {as_of_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
