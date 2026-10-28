"""A-share data adapters backed by AKShare; no changes to Agent contracts."""

from __future__ import annotations

import re
from datetime import date as calendar_date, timedelta

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError

_A_SHARE = re.compile(r"^(\d{6})\.(SH|SS|SZ|BJ)$", re.IGNORECASE)
_STATEMENTS = {
    "balance": "资产负债表",
    "cashflow": "现金流量表",
    "income": "利润表",
}


def _ak():
    try:
        import akshare
    except ImportError as exc:
        raise VendorUnavailableError("AKShare is not installed; install trader[ashare]") from exc
    return akshare


def _call(name: str, **kwargs):
    """Convert vendor HTTP failures to the existing non-fabricating fallback sentinel."""
    try:
        return getattr(_ak(), name)(**kwargs)
    except (NoMarketDataError, VendorUnavailableError):
        raise
    except Exception as exc:
        raise VendorUnavailableError(f"AKShare {name} unavailable: {type(exc).__name__}: {exc}") from exc


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
    """Existing contract: return source-labelled CSV with inclusive analysis dates."""
    df = _price_frame(symbol, start_date, end_date).set_index("Date")
    return (
        f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        f"# Source: Tencent via AKShare; unadjusted, volume in shares\n"
        f"# Total records: {len(df)}\n\n{df.to_csv()}"
    )


def _statement(symbol: str, freq: str, curr_date: str, kind: str) -> str:
    code, exchange = _code(symbol)
    if not curr_date:
        raise ValueError("Financial statements require analysis date")
    if freq not in {"quarterly", "annual"}:
        raise ValueError(f"Unsupported statement frequency: {freq}")
    df = _call("stock_financial_report_sina",
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


def is_ashare(symbol: str) -> bool:
    return isinstance(symbol, str) and _A_SHARE.fullmatch(symbol.strip()) is not None


def _price_frame(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Historical A-share OHLCV, one canonical dataframe for all downstream consumers."""
    code, exchange = _code(symbol)
    df = _call("stock_zh_a_hist_tx",
        symbol=f"{exchange}{code}",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="",
        timeout=15,
    )
    _unavailable(df, symbol, f"No A-share OHLCV for {start_date}..{end_date}")
    needed = ("date", "open", "high", "low", "close", "volume")
    if any(column not in df.columns for column in needed):
        raise ValueError("AKShare Tencent OHLCV missing required columns")
    df = df.copy()
    # Tencent AKShare may leave sz000 volume in lots (100 shares).
    # Use the transaction-value implied mean price to detect the unit, not a blind scale.
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
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume", "amount": "Amount",
    })
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]
    _unavailable(df, symbol, "No usable A-share OHLCV rows in requested window")
    return df.reset_index(drop=True)


def load_ohlcv(symbol: str, as_of_date: str, *, history_days: int = 550) -> pd.DataFrame:
    """Stockstats-ready data for indicators/snapshots, cut off at analysis date."""
    end = pd.Timestamp(as_of_date)
    start = (end - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")
    return _price_frame(symbol, start, as_of_date)


def get_indicators(
    symbol: str, indicator: str, curr_date: str, look_back_days: int = 30
) -> str:
    """Compute current and trailing indicator values using the same A-share OHLCV."""
    from stockstats import wrap

    df = load_ohlcv(symbol, curr_date)
    supported = {
        "close_50_sma", "close_200_sma", "close_10_ema", "macd", "macds",
        "macdh", "rsi", "boll", "boll_ub", "boll_lb", "atr", "vwma", "mfi",
    }
    if indicator not in supported:
        raise ValueError(f"Unsupported indicator: {indicator}")
    stats = wrap(df.copy())
    values = stats[indicator]
    dates = pd.to_datetime(df["Date"])
    cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=look_back_days)
    lines = [
        f"## {indicator} from {cutoff.date()} to {curr_date}",
        "Source: Tencent via AKShare, calculated locally with stockstats",
    ]
    for date, value in zip(dates, values, strict=True):
        if cutoff <= date <= pd.Timestamp(curr_date):
            rendered = "N/A" if pd.isna(value) else f"{float(value):.6g}"
            lines.append(f"{date.date()}: {rendered}")
    return "\n".join(lines)


def company_identity(symbol: str) -> dict[str, str]:
    """Find current identity; official CNINFO name is the backup for Eastmoney failures."""
    code, _ = _code(symbol)
    try:
        df = _call("stock_individual_info_em", symbol=code, timeout=15)
        if df is not None and not df.empty and {"item", "value"}.issubset(df.columns):
            pairs = dict(zip(df["item"], df["value"], strict=True))
            identity = {}
            for source, target in (("股票简称", "company_name"), ("行业", "industry")):
                value = pairs.get(source)
                if pd.notna(value) and str(value).strip():
                    identity[target] = str(value).strip()
            if identity.get("company_name"):
                return identity
    except (NoMarketDataError, VendorUnavailableError, KeyError, ValueError):
        pass

    # Fallback uses the same official announcement index validated in Stage A.
    today = calendar_date.today()
    df = _call(
        "stock_zh_a_disclosure_report_cninfo",
        symbol=code,
        start_date=(today - timedelta(days=365)).strftime("%Y%m%d"),
        end_date=today.strftime("%Y%m%d"),
    )
    if df is None or df.empty or not {"代码", "简称"}.issubset(df.columns):
        raise NoMarketDataError(symbol, symbol, "No verified company identity from CNINFO")
    rows = df[df["代码"].astype(str).eq(code)]
    if rows.empty:
        raise NoMarketDataError(symbol, symbol, "Company code missing from CNINFO")
    name = rows.iloc[0]["简称"]
    if pd.isna(name) or not str(name).strip():
        raise NoMarketDataError(symbol, symbol, "Empty company name in CNINFO")
    return {"company_name": str(name).strip()}


def _dividend_disclosure_evidence(ticker: str, curr_date: str) -> str:
    """Supply dated primary-source dividend evidence, never an estimated payout."""
    end = pd.Timestamp(curr_date)
    start = (end - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    try:
        announcements = _announcement_frame(ticker, start, curr_date)
    except (VendorUnavailableError, NoMarketDataError, ValueError) as exc:
        return f"Cash dividend disclosure: DATA_UNAVAILABLE ({type(exc).__name__})"
    if announcements.empty:
        return "Cash dividend disclosure: DATA_UNAVAILABLE (no dated announcement)"

    titles = announcements["公告标题"].fillna("").astype(str)
    subset = announcements[
        titles.str.contains("权益分派|利润分配|分红派息", regex=True, na=False)
    ].copy()
    if subset.empty:
        return "Cash dividend disclosure: DATA_UNAVAILABLE (no matching announcement)"

    # Prefer the actual implementation over earlier proposals and board resolutions.
    subset["_priority"] = subset["公告标题"].astype(str).str.contains(
        "实施", regex=False
    ).astype(int)
    subset = subset.sort_values(
        ["_priority", "公告时间"], ascending=[False, False]
    )
    row = subset.iloc[0]
    title = str(row["公告标题"])
    day = str(row["公告时间"].date())
    source = str(row["公告链接"])
    try:
        body = _pdf_text_from_announcement(source)
    except Exception as exc:
        return (
            f"Dividend announcement: {day} {title}; source: {source}. "
            f"PDF numbers: DATA_UNAVAILABLE ({type(exc).__name__})."
        )
    # Include a tightly bounded excerpt near the actual dividend term,
    # rather than a long PDF that hides the cash distribution facts in noise.
    markers = ("每10股", "每 10 股", "每股派", "现金股利", "派发现金", "分配方案")
    location = next((body.find(x) for x in markers if x in body), -1)
    if location < 0:
        return (
            f"Dividend announcement: {day} {title}; source: {source}. "
            "PDF extracted but dividend amount not verified."
        )
    excerpt = body[max(0, location - 220) : location + 1100]
    return (
        f"## Official cash dividend evidence (as of {curr_date})\n"
        f"Announcement: {title}\nPublished: {day}\nSource: {source}\n"
        f"Official PDF excerpt: {excerpt}\n"
        "Use only explicitly disclosed figures; units are crucial (per-share vs per-10-shares)."
    )


def get_fundamentals(ticker: str, curr_date: str) -> str:
    """Historical fundamentals grounded in disclosed income and balance-sheet data."""
    sections = []
    try:
        identity = company_identity(ticker)
        if identity.get("company_name"):
            sections.append(f"Company (current name): {identity['company_name']}")
    except Exception:
        sections.append("Company profile: DATA_UNAVAILABLE")
    for kind in ("income", "balance"):
        sections.append(_statement(ticker, "quarterly", curr_date, kind))
    sections.append(_dividend_disclosure_evidence(ticker, curr_date))
    sections.append(
        "Valuation ratios / market capitalization: DATA_UNAVAILABLE; do not estimate. "
        "Historical disclosure cutoff is applied to all statement rows."
    )
    return "\n\n".join(sections)


def _announcement_frame(
    ticker: str, start_date: str, end_date: str, category: str = ""
) -> pd.DataFrame:
    code, _ = _code(ticker)
    df = _call("stock_zh_a_disclosure_report_cninfo",
        symbol=code, start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""), category=category,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    required = {"代码", "公告标题", "公告时间", "公告链接"}
    if not required.issubset(df.columns):
        raise ValueError("CNINFO announcement response missing required fields")
    out = df.copy()
    out["公告时间"] = pd.to_datetime(out["公告时间"], errors="coerce")
    out = out[out["公告时间"].between(
        pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1),
        inclusive="left",
    )]
    return out.sort_values("公告时间", ascending=False)


def _pdf_text_from_announcement(link: str, *, max_pages: int = 8) -> str:
    """Download the official linked PDF only; fail closed if link cannot be resolved."""
    from io import BytesIO
    from urllib.parse import parse_qs, urlparse

    import requests
    from pypdf import PdfReader

    query = parse_qs(urlparse(link).query)
    identifier = query.get("announcementId", [""])[0]
    day = query.get("announcementTime", [""])[0][:10]
    if not (identifier.isdigit() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)):
        raise ValueError("Announcement has no verifiable PDF identifier/date")
    url = f"https://static.cninfo.com.cn/finalpage/{day}/{identifier}.PDF"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.content
    if not data.startswith(b"%PDF-") or len(data) > 15_000_000:
        raise ValueError("Announcement document is not a supported PDF")
    reader = PdfReader(BytesIO(data))
    return "\n".join(
        (page.extract_text() or "") for page in reader.pages[:max_pages]
    )[:9000].strip()


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Company disclosures as dated A-share news evidence, with limited PDF extraction."""
    df = _announcement_frame(ticker, start_date, end_date)
    if df.empty:
        return f"NO_NEWS_AVAILABLE: no company announcements for {ticker} in this window"
    lines = [f"# Official CNINFO announcements for {ticker} ({start_date}..{end_date})"]
    for i, (_, row) in enumerate(df.head(12).iterrows()):
        link = str(row["公告链接"])
        lines.append(
            f"\n## {row['公告时间'].date()}: {row['公告标题']}\nSource: {link}"
        )
        if i < 3:
            try:
                body = _pdf_text_from_announcement(link)
                lines.append(
                    f"PDF extract: {body}" if len(body) >= 50
                    else "PDF text unavailable (scanned or empty document)"
                )
            except Exception as exc:
                lines.append(f"PDF text DATA_UNAVAILABLE ({type(exc).__name__})")
    return "\n".join(lines)


def get_insider_transactions(ticker: str, curr_date: str = None) -> str:
    """Conservative optional response until shareholder transaction data is verified."""
    _code(ticker)
    return (
        "DATA_UNAVAILABLE: independently verified A-share insider-trading series "
        "has not been connected; do not infer insider activity from missing data."
    )


def get_closes(ticker: str, start_date: str, end_date: str) -> pd.Series:
    df = _price_frame(ticker, start_date, end_date)
    return df.set_index("Date")["Close"]
