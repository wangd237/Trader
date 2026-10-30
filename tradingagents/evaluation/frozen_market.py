"""Opt-in frozen OHLCV for the REAL-LLM Market Analyst benchmark.

The CSV is supplied by the experimenter, not bundled or downloaded.
No online fallback is permitted when frozen mode is active.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.market import get_stock_stats_indicators_window

REQUIRED = ("Date", "Open", "High", "Low", "Close", "Volume")


def fixture_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_ohlcv(
    symbol: str, as_of_date: str, *, fill_gaps: bool = True,
) -> pd.DataFrame:
    """Validate CSV, ticker and time coverage; return real, unchanged CSV bars."""
    cfg = get_config()
    path = cfg.get("benchmark_ohlcv_csv")
    if not path:
        raise ValueError("Frozen OHLCV is not configured")
    expected_symbol = cfg.get("benchmark_symbol")
    if not expected_symbol or normalize_symbol(symbol) != normalize_symbol(expected_symbol):
        raise ValueError(f"Frozen market fixture only supports {expected_symbol!r}, not {symbol!r}")

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Frozen OHLCV file missing: {source}")
    expected_digest = cfg.get("benchmark_ohlcv_sha256")
    digest = fixture_sha256(source)
    if not expected_digest or digest != expected_digest:
        raise ValueError("Frozen OHLCV SHA-256 differs from the pinned benchmark fixture")

    df = pd.read_csv(source, dtype={"Date": "string"})
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Frozen OHLCV CSV missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Frozen OHLCV CSV contains no rows")

    parsed_dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any() or parsed_dates.duplicated().any():
        raise ValueError("Frozen OHLCV dates must be unique, valid YYYY-MM-DD values")
    df = df.loc[:, list(REQUIRED)].copy()
    df["Date"] = parsed_dates
    for column in REQUIRED[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(REQUIRED[1:])].isna().any().any():
        raise ValueError("Frozen OHLCV has missing/non-numeric price or volume")
    if not (
        df["Open"].gt(0).all() and df["High"].gt(0).all()
        and df["Low"].gt(0).all() and df["Close"].gt(0).all()
        and df["Volume"].ge(0).all()
    ):
        raise ValueError("Frozen OHLCV contains nonpositive prices or negative volume")
    if not (
        df["High"].ge(df[["Open", "Close", "Low"]].max(axis=1)).all()
        and df["Low"].le(df[["Open", "Close", "High"]].min(axis=1)).all()
    ):
        raise ValueError("Frozen OHLCV violates High/Low price bounds")
    df = df.sort_values("Date").reset_index(drop=True)

    cutoff = pd.Timestamp(as_of_date)
    if cutoff.normalize() != cutoff:
        raise ValueError("Frozen OHLCV analysis cutoff must be a date")
    if df["Date"].min() > cutoff:
        raise NoMarketDataError(symbol, symbol, "frozen data starts after requested date")
    filtered = df.loc[df["Date"].le(cutoff)].copy()
    if filtered.empty or (cutoff - filtered["Date"].max()).days > 7:
        raise NoMarketDataError(symbol, symbol, "frozen data is missing/stale at requested date")
    # Only the past is ever visible to an Agent, even if the fixture has future rows.
    return filtered


def get_frozen_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("Frozen stock-data start_date must not exceed end_date")
    df = load_frozen_ohlcv(symbol, end_date)
    if df["Date"].min() > start:
        raise NoMarketDataError(
            symbol, symbol, f"frozen fixture does not cover the start date {start_date}"
        )
    selected = df.loc[df["Date"].between(start, end)]
    if selected.empty:
        raise NoMarketDataError(symbol, symbol, "no frozen OHLCV rows in window")
    return (
        f"# FROZEN BENCHMARK stock data for {symbol.upper()} "
        f"from {start_date} to {end_date}\n"
        f"# Data source: fixed local CSV, never online\n"
        f"# Total records: {len(selected)}\n\n"
        + selected.to_csv(index=False, date_format="%Y-%m-%d")
    )


def route_frozen_market(method: str, *args, **kwargs) -> str:
    """All Market Analyst price/indicator calls use one frozen data source."""
    if method == "get_stock_data":
        return get_frozen_stock_data(*args, **kwargs)
    if method == "get_indicators":
        # Reuse the existing indicator calculations and text formatting:
        # they consume load_ohlcv(), which switches to this same frozen CSV.
        return get_stock_stats_indicators_window(*args, **kwargs)
    raise RuntimeError(f"Frozen Market Analyst blocks unsupported tool: {method}")
