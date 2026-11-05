"""Hong Kong listed equity OHLCV and indicators from Sina through AKShare.

The symbol is a *Hong Kong* security, e.g. 9988.HK -> 09988 for Sina.
Prices are unadjusted HKD exchange quotes; not US ADRs and not Yahoo's
adjusted-close series. Published dates are screened at the data boundary.
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from threading import Lock

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo.ohlcv import _assert_ohlcv_not_stale

_HK_SHARE = re.compile(r"^(\d{1,5})\.HK$", re.IGNORECASE)
_COLUMNS = ("date", "open", "high", "low", "close", "volume")
_HK_HISTORY_LOCK = Lock()

_INDICATORS = {
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh", "rsi",
    "boll", "boll_ub", "boll_lb", "atr", "vwma", "mfi",
}


def is_hk(symbol: str) -> bool:
    return isinstance(symbol, str) and _HK_SHARE.fullmatch(symbol.strip()) is not None


def _code(symbol: str) -> str:
    if not is_hk(symbol):
        raise ValueError(f"Not a Hong Kong stock ticker: {symbol!r}")
    # AKShare stock_hk_daily requires the five-digit HKEX code,
    # while CLI/Yahoo code can be written with one to five digits.
    return f"{int(_HK_SHARE.fullmatch(symbol.strip()).group(1)):05d}"


def _download(symbol: str) -> pd.DataFrame:
    code = _code(symbol)
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorUnavailableError("HK history requires AKShare (pip install -e '.[ashare]')") from exc
    try:
        raw = ak.stock_hk_daily(symbol=code, adjust="")
    except Exception as exc:
        raise VendorUnavailableError(
            f"Sina HK history unavailable for {symbol}: {type(exc).__name__}: {exc}"
        ) from exc
    if raw is None or raw.empty:
        raise NoMarketDataError(symbol, symbol, "Sina HK history has no bars")
    data = raw.copy()
    if "date" not in data.columns and data.index.name == "date":
        data = data.reset_index()
    missing = set(_COLUMNS) - set(data.columns)
    if missing:
        raise VendorUnavailableError(f"Sina HK history missing OHLCV fields: {sorted(missing)}")
    data = data[list(_COLUMNS)].rename(columns={
        "date": "Date", "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    data = data[
        data["Open"].gt(0) & data["High"].ge(data["Low"])
        & data["Close"].gt(0) & data["Volume"].ge(0)
    ]
    data = data.sort_values("Date").drop_duplicates("Date", keep="last")
    if data.empty:
        raise NoMarketDataError(symbol, symbol, "Sina HK history contains no valid OHLCV")
    return data.reset_index(drop=True)


@lru_cache(maxsize=32)
def _historic_download(symbol: str) -> pd.DataFrame:
    # A single stock_hk_daily call downloads its entire archive, so several
    # indicator calls within the same historical analysis can reuse it.
    return _download(symbol)


def price_frame(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    code = _code(symbol)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date is later than end_date")
    # Do not persist a current or future day's potentially unfinished bar.
    # functools.lru_cache is thread-safe for its internal dictionary, but
    # concurrent *first misses* can still run the wrapped function multiple
    # times. AKShare stock_hk_daily constructs a fresh native V8 MiniRacer for
    # every download; concurrent cold misses can abort Windows Python rather
    # than raise a catchable exception. Serialize the entire native call,
    # including the cache-miss check.
    with _HK_HISTORY_LOCK:
        data = (
            _download(f"{code}.HK")
            if end.date() >= date.today()
            else _historic_download(f"{code}.HK").copy()
        )
    data = data[data["Date"].between(start, end)]
    if data.empty:
        raise NoMarketDataError(symbol, symbol, f"no HK bars for {start_date}..{end_date}")
    _assert_ohlcv_not_stale(data, end_date, symbol, symbol)
    return data.reset_index(drop=True)


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    df = price_frame(symbol, start_date, end_date).set_index("Date")
    return (
        f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        "# Source: Sina Finance HK via AKShare; unadjusted prices in HKD, volume in shares\n"
        f"# Total records: {len(df)}\n\n{df.to_csv()}"
    )


def load_ohlcv(symbol: str, as_of_date: str, *, history_days: int = 550) -> pd.DataFrame:
    end = pd.Timestamp(as_of_date)
    start = (end - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")
    return price_frame(symbol, start, as_of_date)


def get_indicators(
    symbol: str, indicator: str, curr_date: str, look_back_days: int = 30
) -> str:
    if indicator not in _INDICATORS:
        raise ValueError(f"Unsupported indicator: {indicator}")
    df = load_ohlcv(symbol, curr_date)
    stats = wrap(df.copy())
    try:
        values = stats[indicator]
    except (KeyError, TypeError, ValueError) as exc:
        raise VendorUnavailableError(f"Sina HK could not compute {indicator}") from exc
    cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=look_back_days)
    lines = [
        f"## {indicator} from {cutoff.date()} to {curr_date}",
        "Source: Sina Finance HK via AKShare, locally calculated with stockstats",
    ]
    for day, value in zip(df["Date"], values, strict=True):
        if cutoff <= day <= pd.Timestamp(curr_date):
            rendered = "N/A" if pd.isna(value) else f"{float(value):.6g}"
            lines.append(f"{day.date()}: {rendered}")
    return "\n".join(lines)
