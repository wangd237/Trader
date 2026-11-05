"""HKEXnews official announcement *index* (publication dates, titles, original PDF URLs).

HKEX does not publish a stable supported REST contract for titleSearchServlet.
Its site uses a JSON stock-id directory and a double-encoded JSON search
response. Treat missing/malformed responses as unavailable, never as evidence
that an issuer did not publish a filing. No PDFs are needed for the news index.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from urllib.parse import urljoin, urlparse

import requests

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors.hk import _code

_BASE = "https://www1.hkexnews.hk"
_STOCKS = _BASE + "/ncms/script/eds/activestock_sehk_e.json"
_SEARCH = _BASE + "/search/titleSearchServlet.do"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": _BASE + "/search/titlesearch.xhtml?lang=en",
}
_RESULTS = re.compile(
    r"results?|final results|interim results|quarterly results|financial results|"
    r"earnings|业绩|業績|年度報告|年度报告|中期報告|中期报告",
    re.IGNORECASE,
)


def _fetch_json(url: str, *, params: dict | None = None):
    try:
        response = requests.get(url, params=params, headers=_HEADERS, timeout=20)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise VendorUnavailableError(f"HKEXnews request failed: {type(exc).__name__}") from exc


def _stocks(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("stock", "stocks", "data", "result"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
    raise VendorUnavailableError("HKEXnews stock directory has unknown format")


@lru_cache(maxsize=64)
def _stock_id(code: str) -> str:
    # The HKEX API requires the internal 'i' value, not '09988' itself.
    for row in _stocks(_fetch_json(_STOCKS)):
        if not isinstance(row, dict):
            continue
        listed_code = str(row.get("c", "")).strip()
        if listed_code.isdigit() and int(listed_code) == int(code):
            internal = str(row.get("i", "")).strip()
            if internal.isdigit():
                return internal
    raise NoMarketDataError(f"{code}.HK", detail="code not in HKEX active-stock directory")


def _parse_release(text: str) -> datetime | None:
    value = str(text or "").strip()
    for fmt in (
        "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    # A date-only field is insufficient for intraday verification.
    return None


def _result_rows(payload) -> list[dict]:
    if not isinstance(payload, dict) or "result" not in payload:
        raise VendorUnavailableError("HKEXnews search returned no result field")
    inner = payload["result"]
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except ValueError as exc:
            raise VendorUnavailableError("HKEXnews result is not valid nested JSON") from exc
    if not isinstance(inner, list):
        raise VendorUnavailableError("HKEXnews result is not an announcement list")
    if payload.get("hasNextRow") in (True, "true", "True", 1):
        raise VendorUnavailableError("HKEXnews result exceeded one page; cannot trust incomplete index")
    return inner


def _pdf_link(raw: str) -> str:
    link = str(raw or "").strip()
    if not link:
        return ""
    if link.startswith("/listedco/"):
        return urljoin(_BASE, link)
    if link.startswith("listedco/"):
        return urljoin(_BASE + "/", link)
    # Prevent sending the LLM unverified arbitrary domains as official documents.
    if link.startswith("https://"):
        parsed = urlparse(link)
        if parsed.hostname in {"www1.hkexnews.hk", "www.hkexnews.hk"}:
            return link
    return ""


def announcements(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """Return only timestamped original HKEX filings for this stock and interval."""
    code = _code(symbol)
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start > end:
        raise ValueError("start_date is after end_date")
    params = {
        "sortDir": "0",
        "sortByOptions": "DateTime",
        "category": "0",
        "market": "SEHK",
        "stockId": _stock_id(code),
        "documentType": "-1",
        "fromDate": start.strftime("%Y%m%d"),
        "toDate": end.strftime("%Y%m%d"),
        "title": "",
        "searchType": "0",
        "t1code": "-2",
        "t2Gcode": "-2",
        "t2code": "-2",
        "rowRange": "2000",
        "lang": "E",
    }
    rows = _result_rows(_fetch_json(_SEARCH, params=params))
    results: list[dict] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        # HKEX sometimes shows multiple paired securities in STOCK_CODE.
        listed = re.split(r"<br\s*/?>|\s+", str(row.get("STOCK_CODE", "")), flags=re.I)
        if not any(part.isdigit() and int(part) == int(code) for part in listed):
            continue
        published = _parse_release(row.get("DATE_TIME", ""))
        link = _pdf_link(row.get("FILE_LINK", ""))
        title = re.sub(r"<[^>]+>", " ", str(row.get("TITLE", ""))).strip()
        if not published or not link or not title or not (start <= published.date() <= end):
            continue
        key = (link, published)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "title": title,
            "published": published,  # exchange local time, Asia/Hong_Kong
            "url": link,
            "category": str(row.get("LONG_TEXT", "")).strip(),
        })
    return sorted(results, key=lambda x: x["published"], reverse=True)


def result_filings(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """Only official earnings/results announcements; omit generic annual documents."""
    return [
        row for row in announcements(symbol, start_date, end_date)
        if _RESULTS.search(row["title"])
        and not re.search(
            r"date of board meeting|board meeting notice|announcement of meeting|"
            r"change of date|profit warning|profit alert|盈利預警|盈利警告",
            row["title"], re.IGNORECASE,
        )
    ]


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    rows = announcements(ticker, start_date, end_date)
    if not rows:
        raise NoMarketDataError(ticker, ticker, "no timestamped HKEX announcements in window")
    lines = [f"## Official HKEXnews disclosures for {ticker} ({start_date}..{end_date})"]
    for row in rows[:20]:
        lines.append(
            f"### {row['title']}\n"
            f"Published (Asia/Hong_Kong): {row['published']:%Y-%m-%d %H:%M}\n"
            f"Official document: {row['url']}"
        )
    return "\n\n".join(lines)
