"""Point-in-time screened HK financial statements.

Eastmoney provides *current* long-form financial rows keyed by REPORT_DATE,
not their original release dates. A row is exposed only after finding an HKEX
results filing with a verifiable period end (and compatible period length).
This prevents pre-release lookahead, but Eastmoney's values may be restated
after the first publication: original PDF links are retained for audit.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import date, timedelta
from functools import lru_cache
from io import BytesIO

import pandas as pd
import requests

from tradingagents.dataflows.errors import NoMarketDataError, VendorError, VendorUnavailableError
from tradingagents.dataflows.vendors import hk, hkex

_KINDS = {
    "balance": "资产负债表",
    "income": "利润表",
    "cashflow": "现金流量表",
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_EN_MONTH = "|".join(_MONTHS)
_EXACT_EN = re.compile(
    rf"\b(\d{{1,2}})\s+({_EN_MONTH})\s*,?\s*(20\d{{2}})\b|"
    rf"\b({_EN_MONTH})\s+(\d{{1,2}}),?\s*(20\d{{2}})\b",
    re.IGNORECASE,
)
_EXACT_ISO = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")
_EXACT_CN = re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日")
_QUARTER = re.compile(
    rf"\b({_EN_MONTH})\s+quarter\s+(20\d{{2}})\b|"
    rf"\b(20\d{{2}})\s+({_EN_MONTH})\s+quarter\b",
    re.IGNORECASE,
)
_REPORT_MARKER = re.compile(
    r"results?|earnings|financial|业绩|業績|年度報告|年度报告|季度|quarter|"
    r"中期|interim|截至|ended", re.I,
)
_REVISED = re.compile(
    r"clarification|supplement|correction|revision|amendment|"
    r"澄清|補充|补充|更正|修訂|修订",
    re.IGNORECASE,
)


def _date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _period_in_text(text: str) -> tuple[date, int | None] | None:
    """Find a report period, NOT a publication date or unrelated date.

    Only dates near an "ended"/"截至" statement and explicit named quarter
    periods qualify; an otherwise generic "2026 results" does not.
    """
    src = re.sub(r"\s+", " ", text or "")
    for match in _QUARTER.finditer(src):
        if match.group(1):
            month, year = _MONTHS[match.group(1).lower()], int(match.group(2))
        else:
            month, year = _MONTHS[match.group(4).lower()], int(match.group(3))
        return date(year, month, calendar.monthrange(year, month)[1]), 3

    candidates = []
    for m in _EXACT_EN.finditer(src):
        if m.group(1):
            day, month, year = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
        else:
            day, month, year = int(m.group(5)), _MONTHS[m.group(4).lower()], int(m.group(6))
        candidates.append((m.start(), _date(year, month, day)))
    for m in _EXACT_ISO.finditer(src):
        candidates.append((m.start(), _date(*map(int, m.groups()))))
    for m in _EXACT_CN.finditer(src):
        candidates.append((m.start(), _date(*map(int, m.groups()))))

    for offset, day in sorted(candidates):
        if day is None:
            continue
        before = src[max(0, offset - 60):offset].lower()
        if not re.search(
            r"ended\s*(?:on\s*)?$|as\s+at\s*$|截至.{0,10}$|至.{0,5}$|"
            r"ending\s*(?:on\s*)?$|period\s*to\s*$",
            before, re.IGNORECASE,
        ):
            continue
        context = src[max(0, offset - 100):offset + 80].lower()
        if not _REPORT_MARKER.search(context):
            continue
        if re.search(r"three months|3 months|quarter|季度|三個月|三个月", context):
            months = 3
        elif re.search(r"six months|6 months|half.year|interim|半年|中期|六個月|六个月", context):
            months = 6
        elif re.search(r"year ended|12 months|annual|年度|全年|十二個月|十二个月", context):
            months = 12
        else:
            months = None
        return day, months
    return None


def _pdf_first_pages(url: str) -> str:
    """Read at most the first 2 PDF pages to identify the reporting period."""
    if not url.startswith("https://www1.hkexnews.hk/listedco/"):
        return ""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF-") or len(content) > 12_000_000:
            return ""
        reader = PdfReader(BytesIO(content), strict=False)
        return "\n".join(
            (reader.pages[i].extract_text() or "")
            for i in range(min(2, len(reader.pages)))
        )[:12000]
    except Exception:  # noqa: BLE001 — PDF retrieval/parsing is optional evidence
        return ""


@lru_cache(maxsize=32)
def _filings_for_analysis(ticker: str, as_of_date: str) -> tuple[dict, ...]:
    # Historical annual/quarterly reports may have been released well after
    # their period end. Query back 950 calendar days, bounded per issuer.
    end = date.fromisoformat(as_of_date)
    start = end - timedelta(days=950)
    rows = hkex.result_filings(ticker, start.isoformat(), end.isoformat())
    # A date-only trade decision has no intraday timestamp, so same-day
    # announcements are withheld (including those released after the close).
    return tuple(row for row in rows if row["published"].date() < end)


@lru_cache(maxsize=96)
def _source_rows(code: str, kind: str, frequency: str) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorUnavailableError("HK financial statements require AKShare") from exc
    indicator = "年度" if frequency == "annual" else "报告期"
    try:
        data = ak.stock_financial_hk_report_em(
            stock=code, symbol=_KINDS[kind], indicator=indicator,
        )
    except Exception as exc:
        raise VendorUnavailableError(
            f"Eastmoney HK financial data failed: {type(exc).__name__}"
        ) from exc
    if data is None or data.empty:
        raise NoMarketDataError(f"{code}.HK", detail="no HK financial statements")
    required = {
        "SECURITY_CODE", "REPORT_DATE", "STD_ITEM_CODE", "STD_ITEM_NAME", "AMOUNT",
    }
    # The balance sheet is a point-in-time statement. Its Eastmoney response
    # does not contain START_DATE, unlike income and cash-flow statements.
    if kind != "balance":
        required.add("START_DATE")
    if not required.issubset(data.columns):
        raise VendorUnavailableError(
            f"HK financial rows missing fields: {sorted(required - set(data.columns))}"
        )
    data = data.copy()
    data["REPORT_DATE"] = pd.to_datetime(data["REPORT_DATE"], errors="coerce")
    valid = (
        data["SECURITY_CODE"].astype(str).str.zfill(5).eq(code)
        & data["REPORT_DATE"].notna()
        & data["STD_ITEM_NAME"].notna()
    )
    if kind != "balance":
        data["START_DATE"] = pd.to_datetime(data["START_DATE"], errors="coerce")
        valid &= data["START_DATE"].notna()
    data = data[valid]
    if data.empty:
        raise NoMarketDataError(f"{code}.HK", detail="no valid HK financial rows")
    return data


def _duration_months(start: date, end: date) -> int | None:
    # Start of a reporting interval and end of its last calendar month.
    if start.day != 1 or end.day != calendar.monthrange(end.year, end.month)[1]:
        return None
    return (end.year - start.year) * 12 + end.month - start.month + 1


def _evidence_for_period(
    rows: tuple[dict, ...],
    period_end: date,
    months: int | None,
) -> dict | None:
    # Fail closed for uncertain periods: do not infer "June quarter" from a
    # general earnings release, and do not treat a board-meeting notice as results.
    for row in sorted(rows, key=lambda x: x["published"]):
        if _REVISED.search(row["title"]):
            continue
        period = _period_in_text(row["title"])
        if period is None or period[1] is None:
            # A title can contain the exact end date but no period length.
            # Only promote it when the original HKEX PDF supplies a matching
            # dated period and an unambiguous duration.
            pdf_period = _period_in_text(_pdf_first_pages(row["url"]))
            if period is None or (pdf_period and pdf_period[0] == period[0]):
                period = pdf_period
        if period is None or period[0] != period_end:
            continue
        # Do not use a generic "results for date X" to infer a quarterly or
        # annual period. Balance sheets have no START_DATE to cross-check it.
        if period[1] != months:
            continue
        return row
    return None


def _quality_checked_items(group: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list[str]]:
    """Clean only demonstrable data defects; never infer currency or units.

    A repeated code with *different* amounts is ambiguous: suppress the code
    rather than arbitrarily keeping the first result. Financial negatives and
    outliers alone are NOT errors (especially for cash flows and EPS).
    """
    items = group[["STD_ITEM_CODE", "STD_ITEM_NAME", "AMOUNT"]].copy()
    items["STD_ITEM_CODE"] = items["STD_ITEM_CODE"].astype("string").str.strip()
    items["STD_ITEM_NAME"] = items["STD_ITEM_NAME"].astype("string").str.strip()
    items["AMOUNT"] = pd.to_numeric(items["AMOUNT"], errors="coerce")

    valid = (
        items["STD_ITEM_CODE"].notna()
        & items["STD_ITEM_CODE"].ne("")
        & items["STD_ITEM_NAME"].notna()
        & items["STD_ITEM_NAME"].ne("")
        & items["AMOUNT"].map(lambda x: bool(pd.notna(x) and math.isfinite(x)))
    )
    notes = []
    invalid_count = int((~valid).sum())
    if invalid_count:
        notes.append(f"Excluded {invalid_count} items with missing/non-finite amount or identity")
    items = items[valid].copy()

    # Eastmoney's computed balancing field is not a disclosed accounting line.
    # It produced -94513999991.78 for BABA June 2026 despite lacking any
    # identifiable official financial metric. Never feed it to the LLM.
    pseudo = (
        items["STD_ITEM_NAME"].isin(("非运算项目", "非運算項目"))
        | items["STD_ITEM_CODE"].eq("004099999")
    )
    if pseudo.any():
        notes.append(
            f"Excluded {int(pseudo.sum())} non-accounting computed items "
            "(e.g. 004099999 非运算项目)"
        )
        items = items[~pseudo]

    if items.empty:
        return items, notes

    unique_amounts = items.groupby("STD_ITEM_CODE")["AMOUNT"].nunique(dropna=False)
    conflicting = set(unique_amounts.index[unique_amounts.gt(1)])
    if conflicting:
        items = items[~items["STD_ITEM_CODE"].isin(conflicting)]
        notes.append(
            "Excluded conflicting values for source item code(s): "
            + ", ".join(sorted(conflicting))
        )
    repeated = len(items) - len(items.drop_duplicates("STD_ITEM_CODE"))
    if repeated:
        items = items.drop_duplicates("STD_ITEM_CODE", keep="first")
        notes.append(f"Collapsed {repeated} exact same-code duplicate entries")

    # Revenue synonyms in the observed BABA income schema. Suppress only
    # when the source explicitly reports *the same value* for both codes.
    if kind == "income":
        primary = items.loc[items["STD_ITEM_CODE"].eq("004001001"), "AMOUNT"]
        alias = items.loc[items["STD_ITEM_CODE"].eq("004001999"), "AMOUNT"]
        if not primary.empty and not alias.empty:
            if float(primary.iloc[0]) == float(alias.iloc[0]):
                items = items[~items["STD_ITEM_CODE"].eq("004001999")]
                notes.append(
                    "Revenue alias 004001999 (营运收入) suppressed: identical to "
                    "004001001 (营业额); do not double-count"
                )
            else:
                notes.append(
                    "Revenue labels 004001001 and 004001999 differ: "
                    "retain both and verify accounting definitions before combining"
                )
    return items, notes


def _statement(ticker: str, kind: str, frequency: str, curr_date: str) -> str:
    code = hk._code(ticker)
    if frequency not in {"quarterly", "annual"}:
        raise ValueError("freq must be quarterly or annual")
    if not curr_date:
        raise ValueError("HK statements require an analysis date")
    analysis_day = date.fromisoformat(curr_date)
    rows = _source_rows(code, kind, frequency)
    rows = rows[rows["REPORT_DATE"].dt.date < analysis_day]
    if rows.empty:
        raise NoMarketDataError(ticker, detail="no HK financial period before analysis date")
    evidence = _filings_for_analysis(f"{code}.HK", curr_date)
    if not evidence:
        raise NoMarketDataError(ticker, detail="no dated official HKEX results announcements")

    dates = sorted(rows["REPORT_DATE"].dt.date.unique(), reverse=True)
    sections = []
    for report_day in dates:
        block = rows[rows["REPORT_DATE"].dt.date == report_day]
        groups = []
        if kind == "balance":
            # Balance sheet values are measured *at* REPORT_DATE, not over an
            # income/cash-flow interval. Use the actual HKEX results filing
            # to identify the annual vs quarterly/interim disclosure context.
            allowed_lengths = (12,) if frequency == "annual" else (3, 6)
            matched = [
                (months, filing)
                for months in allowed_lengths
                if (filing := _evidence_for_period(evidence, report_day, months))
                is not None
            ]
            # Reject ambiguous matches instead of inventing a START_DATE.
            if len(matched) == 1:
                months, filing = matched[0]
                groups.append((None, months, block, filing))
        else:
            for start in sorted(block["START_DATE"].dt.date.unique(), reverse=True):
                group = block[block["START_DATE"].dt.date == start]
                months = _duration_months(start, report_day)
                if months not in {3, 6, 12}:
                    continue
                if frequency == "annual" and months != 12:
                    continue
                if frequency == "quarterly" and months == 12:
                    continue
                filing = _evidence_for_period(evidence, report_day, months)
                if filing is not None:
                    groups.append((start, months, group, filing))

        for start, months, group, filing in groups:
            show, quality_notes = _quality_checked_items(group, kind)
            if show.empty:
                continue
            heading = (
                f"### Balance sheet as of {report_day} "
                f"(disclosed with {months}-month results)"
                if start is None else
                f"### Report period {start} to {report_day} ({months} months)"
            )
            lines = [
                heading,
                f"First verified HKEX publication: {filing['published']:%Y-%m-%d %H:%M} (Asia/Hong_Kong)",
                f"Official original filing: {filing['url']}",
                f"Source of numeric financial rows: Eastmoney HK via AKShare ({_KINDS[kind]})",
                "Currency/amount units: UNVERIFIED. Financial reporting currency need not be HKD. "
                "All AMOUNT figures remain unscaled, as returned by Eastmoney.",
                "EPS/per-share fields (if present) are per-share amounts; do not add them to "
                "monetary financial-statement totals.",
                "Quality checks: " + ("; ".join(quality_notes) if quality_notes else
                                     "no invalid, conflicting or placeholder rows detected"),
                f"Usable source items: {len(show)}; displaying up to 80 rows.",
                show.head(80).to_csv(index=False),
            ]
            sections.append("\n".join(lines))
            if len(sections) == 2:
                break
        if len(sections) == 2:
            break
    if not sections:
        raise NoMarketDataError(
            ticker, detail=f"no {kind} report has a verified as-of-date HKEX disclosure"
        )
    return (
        f"## {ticker} HK financial statement: {_KINDS[kind]}\n"
        f"Analysis date: {curr_date}. Same-day filings are excluded.\n"
        "CAVEAT: Eastmoney rows reflect its current database and may incorporate later "
        "revisions; the linked original HKEX filing is the as-reported evidence. "
        "Do not claim the current numeric values are immutable point-in-time snapshots.\n\n"
        + "\n\n".join(sections)
    )


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _statement(ticker, "balance", freq, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _statement(ticker, "income", freq, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _statement(ticker, "cashflow", freq, curr_date)


def get_fundamentals(ticker: str, curr_date: str) -> str:
    sections = [f"## Hong Kong fundamentals for {ticker} as of {curr_date}"]
    successes = 0
    for kind in ("income", "balance", "cashflow"):
        try:
            sections.append(_statement(ticker, kind, "quarterly", curr_date))
            successes += 1
        except VendorError as exc:
            sections.append(f"{_KINDS[kind]}: DATA_UNAVAILABLE ({exc})")
    if not successes:
        raise NoMarketDataError(ticker, detail="no point-in-time screened HK fundamental reports")
    sections.append(
        "Valuation/multiples: DATA_UNAVAILABLE unless independently verified as of "
        "analysis date. Do not invent company valuation from missing data."
    )
    return "\n\n".join(sections)
