"""从 SEC EDGAR 获取公司财报，严格按照实际披露日期提供历史版本。

其他基本面接口常按财务期末归档当前数值，但期末不代表已经公开，
而且之后重述的数字也不是投资者当时看到的数据。
EDGAR 为每个财务事实记录披露日期，因此历史 ``as_of_date``
只能获取彼时已提交的资料；修订值从修订披露日才开始生效。
例如苹果 2008 年总资产原披露约 396 亿美元，2010 年修订为
362 亿美元，在修订前的时点不能提前使用后者。

无需账户或 API Key，但 SEC 要求明确标识调用方的 User-Agent；
仅覆盖美国申报公司。ticker 不在 EDGAR 映射表时，
路由层交由下一个已配置提供商处理。"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import requests

from tradingagents import __version__
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.files import replace_file

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# 申报历史只会在有新文件提交时变化，
# 因此同一公司每天抓取一次即可服务当天所有历史日期请求。
_CACHE_TTL_SECONDS = 24 * 60 * 60

# 同一个财务项目可能对应多个申报标签，按优先级匹配，首个命中生效。
# 不同标签的值不能相加，否则同一收入被多个标签表示时会重复计算。
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets",)),
        ("Current Assets", ("AssetsCurrent",)),
        ("Cash and Equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
        ("Total Liabilities", ("Liabilities",)),
        ("Current Liabilities", ("LiabilitiesCurrent",)),
        ("Stockholders Equity", ("StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ],
    "income_statement": [
        ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "SalesRevenueNet")),
        ("Cost of Revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        ("Gross Profit", ("GrossProfit",)),
        ("Operating Income", ("OperatingIncomeLoss",)),
        ("Net Income", ("NetIncomeLoss",)),
        ("Diluted EPS", ("EarningsPerShareDiluted",)),
    ],
    "cashflow": [
        ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",
                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
        ("Investing Cash Flow", ("NetCashProvidedByUsedInInvestingActivities",)),
        ("Financing Cash Flow", ("NetCashProvidedByUsedInFinancingActivities",)),
        ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets")),
    ],
}

# 财报事实有时间跨度：季度约 90 天，年度约 365 天。
# 同一次披露可能包含单季度与年初至今数值，因此不能仅按期末日期匹配，
# 否则可能把半年累计收入当成单季度收入。
_SPANS = {"quarterly": (60, 115), "annual": (300, 400)}

# 10-Q 现金流常只提供年初至今的累计数据。
# 季度报表应优先显示独立季度数；不存在时保留累计数并在列名注明跨度。
# 不通过两次披露相减构造公司未实际披露的季度值。
_YEAR_TO_DATE = ((150, 200, 6), (240, 290, 9))

# 财政年度应与年报覆盖期间一致。
# 10-Q 资产负债时点数没有跨度可检查，某些 10-Q 也包含 12 个月累计值，
# 若仅看跨度可能误当年报。不同表单中更晚披露的重编数据
# 仍应从该披露日起参与历史结果。
_ANNUAL_FORMS = ("10-K", "20-F", "40-F")


def _user_agent() -> str:
    """设置 SEC 识别的调用方身份，无需专门账号或密钥。

    sec.gov 要求 User-Agent 包含联系邮箱，只有名称或项目网址
    可能返回 403。默认使用占位邮箱及包版本。
    用户应通过 SEC_EDGAR_USER_AGENT 填写真实名称和联系方式，
    便于 SEC 就请求行为与调用方联系。"""
    configured = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    return configured or f"TradingAgents/{__version__} (contact@example.com)"




def _fetch_json(url: str) -> dict:
    """遵循 SEC 调用方身份规范读取公开 EDGAR 文档。"""
    try:
        response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # 本模块中的各类失败都表示此提供商当前无法提供数据，
        # 应交由路由器尝试下一来源，避免泄漏没有统一处理的传输异常。
        raise VendorUnavailableError(f"SEC EDGAR request failed ({status or type(exc).__name__})") from exc
    except ValueError as exc:
        raise VendorUnavailableError("SEC EDGAR returned an unreadable response") from exc


def _cached_json(url: str, name: str) -> dict:
    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / name
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass  # a truncated file is a miss, not a failure
    data = _fetch_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, lambda temp: Path(temp).write_text(json.dumps(data), encoding="utf-8"))
    return data


def cik_for(ticker: str) -> str | None:
    """返回申报公司的 CIK；若 ticker 不是美国申报公司，则返回 None。"""
    table = _cached_json(_TICKERS_URL, "company_tickers.json")
    wanted = ticker.strip().upper()
    for entry in table.values():
        if entry.get("ticker", "").upper() == wanted:
            return f"{int(entry['cik_str']):010d}"
    return None


def _span_index(fact: dict, spans: tuple[tuple[int, int], ...]) -> int | None:
    """判断期间型财务事实属于 ``spans`` 中的哪个时间跨度；
    时点型事实返回跨度 0，无法匹配时返回 None。"""
    if "start" not in fact:
        return 0
    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    return next((i for i, (low, high) in enumerate(spans) if low <= days <= high), None)


def _as_of(facts: dict, tags: tuple[str, ...], as_of_date: str, spans: tuple[tuple[int, int], ...],
           forms: tuple[str, ...] = ()) -> tuple[dict, str]:
    """返回申报公司在历史时点已披露的财务事实：
    ``({(期末日期, 跨度索引): 数值}, 单位)``。

    同一期间被多次申报时，选用 as_of_date 之前最新一次申报；
    修订只能从实际披露日起生效。
    单位由申报文件决定：多数项目是 USD，每股收益是 USD/shares，
    不能按同一倍率处理，否则可能把真实每股收益错误显示为零。
    收入、现金流等期间型项目必须匹配给定跨度；
    资产负债表的时点型数据不要求跨度匹配。"""
    values: dict[tuple[str, int], float] = {}
    chosen_unit = "USD"
    # 按顺序尝试财务标签，某个期间保留首个有值的标签。
    # 公司可能在不同年份改用不同名称，所以单个标签不能覆盖全部历史。
    # 禁止把多个标签的数字相加，以免重复计数。
    for tag in tags:
        for unit, unit_values in ((facts.get(tag) or {}).get("units", {})).items():
            latest: dict[tuple[str, int], dict] = {}
            covered: set[tuple[str, int]] = set()   # periods a filing of ``forms`` reports
            for fact in unit_values:
                index = _span_index(fact, spans)
                key = (fact["end"], index)
                if fact["filed"] > as_of_date or index is None or key in values:
                    continue
                if not forms or fact.get("form", "").startswith(forms):
                    covered.add(key)
                seen = latest.get(key)
                if seen is None or fact["filed"] >= seen["filed"]:
                    latest[key] = fact
            latest = {key: fact for key, fact in latest.items() if key in covered}
            if latest:
                chosen_unit = unit
                values.update({key: fact["val"] for key, fact in latest.items()})
    return dict(sorted(values.items())), chosen_unit


def _statement(kind: str, ticker: str, freq: str, as_of_date: str, title: str) -> str:
    as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    cik = cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")

    facts = _cached_json(_FACTS_URL.format(cik=cik), f"CIK{cik}.json")
    us_gaap = (facts.get("facts") or {}).get("us-gaap")
    if not us_gaap:
        raise NoMarketDataError(ticker, ticker, "US filer with no us-gaap facts")

    quarterly = freq.lower() == "quarterly"
    if quarterly:
        spans = (_SPANS["quarterly"], *((low, high) for low, high, _ in _YEAR_TO_DATE))
        names = ["", *(f" ({months} months)" for _, _, months in _YEAR_TO_DATE)]
    else:
        spans, names = (_SPANS["annual"],), [""]
    forms = () if quarterly else _ANNUAL_FORMS
    lines = {label: _as_of(us_gaap, tags, as_of_date, spans, forms) for label, tags in _STATEMENTS[kind]}
    # 每个期间的报表行优先选用可得到的最短数据跨度；
    # 同一列只对应一个期间和一种跨度。
    # 当某行只有累计值而其他行有季度值时，
    # 其累计值也应在列中明确标识，不能混淆口径。
    chosen = {label: {} for label in lines}
    for label, (values, _) in lines.items():
        for end, index in values:
            chosen[label][end] = min(index, chosen[label].get(end, index))
    periods = sorted({(end, index) for spans_of in chosen.values() for end, index in spans_of.items()})
    if not periods:
        raise NoMarketDataError(ticker, ticker, f"no {freq} {title.lower()} filed by {as_of_date}")

    header = (
        f"# {title} for {ticker.upper()} ({freq}), USD in millions unless the row says otherwise\n"
        f"# SEC EDGAR facts filed on or before {as_of_date}, at the values filed then\n\n"
    )
    rows = [",".join([""] + [end + names[index] for end, index in periods])]
    for label, (values, unit) in lines.items():
        # 每一行必须使用相同的列布局，避免读者错误对齐数据。
        if not values:
            rows.append(",".join([label] + ["unavailable (not tagged by this filer)"] * len(periods)))
            continue
        name = label if unit == "USD" else f"{label} ({unit})"
        # 输出普通数字，不添加千位分隔符，避免破坏 CSV 字段结构。
        cells = []
        for end, index in periods:
            value = values.get((end, index)) if chosen[label].get(end) == index else None
            cells.append("" if value is None else f"{value / 1e6:.0f}" if unit == "USD" else f"{value:.2f}")
        rows.append(",".join([name] + cells))
    return header + "\n".join(rows) + "\n"


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """返回不晚于 ``as_of_date`` 实际披露的资产负债表。"""
    return _statement("balance_sheet", ticker, freq, as_of_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """返回不晚于 ``as_of_date`` 实际披露的利润表。

    不能自行推导第四季度：公司往往只在年度数据中包含第四季度；
    用年度数减去其他三个季度会生成一个从未独立披露、
    也没有对应披露日期的数值。"""
    return _statement("income_statement", ticker, freq, as_of_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """返回不晚于 ``as_of_date`` 实际披露的现金流量表。"""
    return _statement("cashflow", ticker, freq, as_of_date, "Cash Flow Statement")
