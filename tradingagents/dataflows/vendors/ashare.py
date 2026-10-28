"""A-share data adapters backed by AKShare; no changes to Agent contracts."""

from __future__ import annotations

import re

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError

_A_SHARE = re.compile(r"^(\d{6})\.(SH|SS|SZ|BJ)$", re.IGNORECASE)
_STATEMENTS = {
    "balance": "资产负债表",
    "cashflow": "现金流量表",
    "income": "利润表",
}


def _ak():
    import akshare
    return akshare


def _code(symbol: str) -> tuple[str, str]:
    match = _A_SHARE.fullmatch(symbol.strip())
    if not match:
        raise ValueError(f"Not an A-share ticker: {symbol!r}")
    code, suffix = match.groups()
    exchange = "sh" if suffix.upper() in {"SH", "SS"} else suffix.lower()
    return code, exchange


def _unavailable(df: pd.DataFrame, symbol: str, detail: str):
    if df is None or df.empty:
        raise NoMarketDataError(symbol, symbol, detail)
    return df


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Existing get_stock_data contract: inclusive YYYY-MM-DD dates, CSV response."""
    code, exchange = _code(symbol)
    df = _ak().stock_zh_a_hist_tx(
        symbol=f"{exchange}{code}",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="",
        timeout=15,
    )
    _unavailable(df, symbol, f"No A-share OHLCV for {start_date}..{end_date}")
    needed = ("date", "open", "high", "low", "close", "volume")
    if any(column not in df.columns for column in needed):
        raise ValueError("AKShare Tencent OHLCV is missing required columns")
    df = df.copy()
    # AKShare historically excludes sz000 codes from its hands-to-shares conversion.
    # Verify by volume/turnover implied price before applying the targeted correction.
    if exchange == "sz" and code.startswith("000") and "amount" in df.columns:
        valid = df["volume"].gt(0) & df["close"].gt(0) & df["amount"].gt(0)
        if valid.any():
            ratio = (
                df.loc[valid, "amount"] / df.loc[valid, "volume"] / df.loc[valid, "close"]
            ).median()
            if 50 <= ratio <= 150:
                df["volume"] *= 100
            elif not 0.5 <= ratio <= 1.5:
                raise ValueError(f"Unexpected Tencent volume unit ratio: {ratio:.3f}")
    rename = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "amount": "Amount",
    }
    df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    _unavailable(df, symbol, "No valid A-share OHLCV rows")
    df = df.set_index("Date")
    return (
        f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        f"# Source: Tencent via AKShare; unadjusted prices, volume in shares\n"
        f"# Total records: {len(df)}\n\n{df.to_csv()}"
    )


def _statement(symbol: str, freq: str, curr_date: str, kind: str) -> str:
    code, exchange = _code(symbol)
    if not curr_date:
        raise ValueError("Financial statements require analysis date")
    if freq not in {"quarterly", "annual"}:
        raise ValueError(f"Unsupported statement frequency: {freq}")
    df = _ak().stock_financial_report_sina(
        stock=f"{exchange}{code}", symbol=_STATEMENTS[kind]
    )
    _unavailable(df, symbol, f"No {_STATEMENTS[kind]} rows")
    for col in ("报告日", "公告日期"):
        if col not in df.columns:
            raise ValueError(f"Missing disclosure time column {col}")
    df = df.copy()
    df["_report"] = pd.to_datetime(df["报告日"].astype(str), errors="coerce")
    df["_published"] = pd.to_datetime(df["公告日期"].astype(str), errors="coerce")
    cutoff = pd.Timestamp(curr_date)
    df = df[df["_published"].notna() & df["_published"].le(cutoff)]
    if freq == "annual":
        df = df[df["_report"].dt.month.eq(12) & df["_report"].dt.day.eq(31)]
    df = df.sort_values("_report", ascending=False).head(4)
    _unavailable(df, symbol, f"No {_STATEMENTS[kind]} disclosed by {curr_date}")
    df = df.drop(columns=["_report", "_published"])
    return (
        f"# {symbol} {_STATEMENTS[kind]} (source: Sina Finance via AKShare)\n"
        f"# Only disclosed on or before {curr_date}; currency/units as returned by source\n\n"
        + df.to_csv(index=False)
    )


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _statement(ticker, freq, curr_date, "balance")


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _statement(ticker, freq, curr_date, "cashflow")


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _statement(ticker, freq, curr_date, "income")
