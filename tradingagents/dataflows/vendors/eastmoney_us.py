"""BABA NYSE daily history from Eastmoney through the existing AKShare extra.

This is a narrow alternative to Yahoo for the currently evaluated ADR.  Do not
guess Eastmoney market prefixes for arbitrary US symbols: 105 is NASDAQ and
106 is NYSE, and a wrong prefix may silently return the wrong instrument.
"""
from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo.ohlcv import _assert_ohlcv_not_stale

logger = logging.getLogger(__name__)

_SYMBOLS = {"BABA": "106.BABA"}
_COLUMNS = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
}


def supports(symbol: str) -> bool:
    return isinstance(symbol, str) and symbol.strip().upper() in _SYMBOLS


def _canonical(symbol: str) -> tuple[str, str]:
    ticker = symbol.strip().upper()
    if ticker not in _SYMBOLS:
        raise NoMarketDataError(symbol, ticker, "not mapped to an Eastmoney US exchange code")
    return ticker, _SYMBOLS[ticker]


def _download(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker, eastmoney_code = _canonical(symbol)
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorUnavailableError("Eastmoney US history requires trader[ashare] / AKShare") from exc

    try:
        raw = ak.stock_us_hist(
            symbol=eastmoney_code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="",  # raw listed ADR daily bars; no hidden corporate-action adjustment
        )
    except Exception as exc:
        raise VendorUnavailableError(
            f"Eastmoney US history failed for {ticker}: {type(exc).__name__}"
        ) from exc

    if raw is None or raw.empty:
        raise NoMarketDataError(symbol, ticker, "Eastmoney returned no daily prices")
    missing = set(_COLUMNS) - set(raw.columns)
    if missing:
        raise VendorUnavailableError(f"Eastmoney US history missing OHLCV fields: {sorted(missing)}")

    data = raw.rename(columns=_COLUMNS)[list(_COLUMNS.values())].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    data = data[
        data["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & data["Open"].gt(0) & data["Close"].gt(0)
        & data["High"].ge(data["Low"]) & data["Volume"].ge(0)
    ]
    data = data.sort_values("Date").drop_duplicates("Date", keep="last")
    if data.empty:
        raise NoMarketDataError(symbol, ticker, "no valid bars in requested date range")
    _assert_ohlcv_not_stale(data, end_date, symbol, ticker)
    return data.reset_index(drop=True)


@lru_cache(maxsize=32)
def _historical_frame(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Reuse the same historical OHLCV within one Python process across the
    # market analyst's multiple indicator/snapshot tool calls.
    return _download(ticker, start_date, end_date)


def price_frame(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker, _ = _canonical(symbol)
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date is later than end_date")
    # Do not retain intraday/present-day bars in the historical in-process cache.
    if pd.Timestamp(end_date).date() >= datetime.now().date():
        return _download(ticker, start_date, end_date)
    return _historical_frame(ticker, start_date, end_date).copy()


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    data = price_frame(symbol, start_date, end_date)
    return (
        f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        "# Source: Eastmoney US via AKShare; prices in USD, unadjusted, volume in shares\n"
        f"# Total records: {len(data)}\n\n"
        + data.set_index("Date").to_csv()
    )


def load_ohlcv(symbol: str, as_of_date: str, history_days: int = 550) -> pd.DataFrame:
    end = pd.Timestamp(as_of_date)
    start = (end - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")
    data = price_frame(symbol, start, as_of_date)
    # A second cutoff protects against a vendor unexpectedly returning later rows.
    return data[data["Date"] <= end].copy()


def get_indicators(
    symbol: str, indicator: str, as_of_date: str, look_back_days: int = 30
) -> str:
    # Keep the public indicator vocabulary and calculate locally with stockstats.
    supported = {
        "close_50_sma", "close_200_sma", "close_10_ema", "macd", "macds",
        "macdh", "rsi", "boll", "boll_ub", "boll_lb", "atr", "vwma", "mfi",
    }
    if indicator not in supported:
        raise ValueError(f"Unsupported indicator: {indicator}")
    data = load_ohlcv(symbol, as_of_date)
    stats = wrap(data.copy())
    try:
        values = stats[indicator]
    except (KeyError, ValueError, TypeError) as exc:
        raise VendorUnavailableError(
            f"Could not calculate {indicator} from Eastmoney OHLCV"
        ) from exc
    dates = data["Date"]
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=look_back_days)
    lines = [
        f"## {indicator} from {cutoff.date()} to {as_of_date}",
        "Source: Eastmoney US via AKShare, unadjusted prices; locally computed with stockstats",
    ]
    for day, value in zip(dates, values, strict=True):
        if cutoff <= day <= pd.Timestamp(as_of_date):
            rendered = "N/A" if pd.isna(value) else f"{float(value):.6g}"
            lines.append(f"{day.date()}: {rendered}")
    return "\n".join(lines)
