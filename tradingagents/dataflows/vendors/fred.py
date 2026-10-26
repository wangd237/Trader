"""FRED（美联储经济数据）宏观数据提供商。

通过圣路易斯联储免费 API 获取政策利率、国债收益率、通胀、
劳动力及经济增长时间序列，供新闻分析师依据真实数字讨论宏观环境。
免费 API Key 通过 ``FRED_API_KEY`` 提供；未设置时抛出
``FredNotConfiguredError``，由路由层按提供商不可用处理。"""
import logging
import os
from datetime import datetime, timedelta

import pytz

from tradingagents.dataflows.errors import VendorNotConfiguredError
from tradingagents.dataflows.net import get_scrubbed

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred"

# FRED 实时日期按美国中部时区计算，未来日期会返回 400。
# 因此版本日期以 FRED 当地今天为上限，而非调用方本地时间（#1275）。
# 依赖中的 pytz 自带时区库，不要求系统安装 tzdata。
FRED_TZ = pytz.timezone("America/Chicago")

# 网络超时秒数，防止请求卡住 Agent，与 Alpha Vantage 客户端一致。
REQUEST_TIMEOUT = 30

# 未指定回溯窗口时采用的默认时长。
# 一年通常能覆盖月度或季度指标的走势及同比比较基期。
DEFAULT_LOOKBACK_DAYS = 365

# 报告表格行数上限：近期数据对决策更重要，
# 限制长窗口内每日国债收益率或 VIX 数据占用的上下文。
MAX_ROWS = 40

# 维护易读别名到 FRED ID 的映射表。列表之外仍允许使用原始 ID，
# 不限制熟悉 FRED 的用户。
MACRO_SERIES = {
    # 政策利率与美国国债收益率。
    "fed_funds_rate": "FEDFUNDS",
    "federal_funds_rate": "FEDFUNDS",
    "fed_funds": "FEDFUNDS",
    "2y_treasury": "DGS2",
    "10y_treasury": "DGS10",
    "30y_treasury": "DGS30",
    "10y_2y_spread": "T10Y2Y",
    "yield_curve": "T10Y2Y",
    # 通货膨胀指标。
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "inflation_expectations": "T10YIE",
    # 经济增长与产出。
    "real_gdp": "GDPC1",
    "gdp": "GDP",
    "industrial_production": "INDPRO",
    # 劳动力市场。
    "unemployment_rate": "UNRATE",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "payrolls": "PAYEMS",
    "initial_claims": "ICSA",
    # 货币与金融市场。
    "m2": "M2SL",
    "money_supply": "M2SL",
    "vix": "VIXCLS",
    "dollar_index": "DTWEXBGS",
    # 消费者信心与住房市场。
    "consumer_sentiment": "UMCSENT",
    "housing_starts": "HOUST",
    "retail_sales": "RSAFS",
    # 欧元区数据（例如欧洲上市的 .PA 标的）：FRED 同时收录
    # 欧洲央行利率、欧盟统计局 HICP/GDP 及 OECD 长期国债收益率。
    # 同样通过固定数据版本日期避免历史信息泄漏。
    "ecb_deposit_rate": "ECBDFR",
    "ecb_main_refi_rate": "ECBMRRFR",
    "euro_hicp": "CP0000EZ19M086NEST",
    "euro_core_hicp": "00XEFDEZ19M086NEST",
    "euro_real_gdp": "CLVMNACSCAB1GQEA19",
    "germany_10y": "IRLTLT01DEM156N",
    "france_10y": "IRLTLT01FRM156N",
    "eur_usd": "DEXUSEU",
}


class FredNotConfiguredError(VendorNotConfiguredError):
    """选择 FRED 却未配置密钥时抛出的异常。
    继承 VendorNotConfiguredError（也是 ValueError），
    兼容现有路由降级和旧版错误捕获逻辑。"""


def get_api_key() -> str:
    """从环境变量获取 FRED API Key。"""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise FredNotConfiguredError(
            "FRED_API_KEY environment variable is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html."
        )
    return api_key


def _resolve_series_id(indicator: str) -> str:
    """将易读别名映射为 FRED 序列 ID；已有合法 ID 则直接透传。

    既不属于已知别名又不符合 FRED ID 形式时抛出 ValueError，
    例如模型给出 "bank of japan rate" 这样的描述性短语。
    提前反馈可避免让非法参数请求 FRED 并得到 HTTP 400。"""
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MACRO_SERIES:
        return MACRO_SERIES[key]
    candidate = indicator.strip().upper()
    # FRED 序列 ID 较短且不含空白字符，
    # 应拒绝 LLM 传入的自然语言短语，而不是直接请求并得到 400。
    if not candidate or len(candidate) > 30 or any(c.isspace() for c in candidate):
        raise ValueError(
            f"'{indicator}' is not a known macro alias or a valid FRED series ID. "
            f"Use an alias (e.g. 'cpi', 'unemployment', '10y_treasury') or a raw "
            f"FRED series ID (e.g. 'CPIAUCSL')."
        )
    return candidate


def _fred_today() -> str:
    """返回 FRED 所在美国中部时区今天的日期，格式 ``yyyy-mm-dd``。

    数据版本日期不能晚于该日期，因为 FRED 会拒绝未来的 realtime_start。
    实时运行中调用者当地可能已经是第二天，不能直接使用本地日期。"""
    return datetime.now(FRED_TZ).strftime("%Y-%m-%d")


def _request(path: str, params: dict) -> dict:
    """GET 请求 FRED 端点；请求失败时提供 FRED JSON 错误详情。"""
    api_key = get_api_key()
    response = get_scrubbed(
        f"{FRED_API_BASE}/{path}",
        params={**params, "api_key": api_key, "file_type": "json"},
        timeout=REQUEST_TIMEOUT,
        secret=api_key,
        passthrough=(400,),
    )
    # 未知序列 ID 或无效请求参数时，FRED 返回含 error_message 的 JSON 400；
    # 将其转换成清晰、可操作的错误。
    if response.status_code == 400:
        try:
            message = response.json().get("error_message", response.text)
        except ValueError:
            message = response.text
        raise ValueError(f"FRED request failed: {message}")
    return response.json()


def get_macro_data(
    indicator: str,
    as_of_date: str,
    look_back_days: int | None = None,
) -> str:
    """获取指定 FRED 宏观时间序列，格式化为 Markdown 报告。

    参数：
        indicator：易读别名（如 cpi、unemployment、10y_treasury）
            或原始 FRED ID（如 CPIAUCSL、DGS10）。
        as_of_date：历史时点，格式 yyyy-mm-dd。用于限制观测窗口，
            同时将 FRED 的 realtime 版本日期固定在该日（不晚于 FRED 今天），
            确保回测取得当时发布的数据，而非后来修订的 CPI/GDP（#1275）。
        look_back_days：回溯天数；None 时使用 DEFAULT_LOOKBACK_DAYS。

    返回：
        Markdown 报告，包含序列标题、单位、频率、最新值、
        窗口内变化及近期数据表。"""
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    # 明确固定数据的历史版本。FRED 默认使用今天的 realtime 窗口，
    # 这会返回各条观测值的最新版，而不是历史日期当时已知的版本。
    # 必须对元数据和观测数据请求都将 realtime 边界设置为 as_of_date（#1275）。
    # 如果调用者当地比美国中部早进入次日，应限制到 FRED 的今天，
    # 防止未来日期触发 400，导致路由器丢失宏观数据。
    # 历史 as_of_date 不受此修正影响。
    pit = min(as_of_date, _fred_today())
    realtime = {"realtime_start": pit, "realtime_end": pit}

    # LLM 传入非法指标时，优先返回具体使用指引而非直接抛异常，
    # 避免坏参数终止运行；宏观数据虽可降级，明确错误更有帮助。
    try:
        series_id = _resolve_series_id(indicator)
    except ValueError as e:
        return f"FRED: {e}"

    meta = _request("series", {"series_id": series_id, **realtime}).get("seriess") or []
    if not meta:
        return (
            f"FRED series '{series_id}' not found. Pass a known alias "
            f"(e.g. 'cpi', 'unemployment') or a valid FRED series ID."
        )
    info = meta[0]
    title = info.get("title", series_id)
    units = info.get("units_short") or info.get("units", "")
    frequency = info.get("frequency", "")
    seasonal = info.get("seasonal_adjustment_short", "")

    observations = _request(
        "series/observations",
        {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": as_of_date,
            "sort_order": "asc",
            **realtime,
        },
    ).get("observations", [])

    # FRED 使用 "." 表示缺失的观测值。
    points = [
        (o["date"], o["value"])
        for o in observations
        if o.get("value") not in (".", None, "")
    ]

    header = (
        f"## FRED: {title} ({series_id})\n"
        f"- Units: {units}\n"
        f"- Frequency: {frequency}"
        f"{f' ({seasonal})' if seasonal else ''}\n"
        f"- Window: {start_date} to {as_of_date}\n"
    )

    if not points:
        return header + (
            f"\nNo observations for {series_id} in this window at the {pit} "
            f"vintage. The series may report less frequently than the window "
            f"(try a longer look_back_days), or have no vintage published by "
            f"then (unpublished as of {pit}, or before ALFRED coverage begins)."
        )

    first_date, first_val = points[0]
    last_date, last_val = points[-1]
    try:
        delta = float(last_val) - float(first_val)
        base = float(first_val)
        pct = f" ({delta / base * 100:+.2f}%)" if base != 0 else ""
        summary = (
            f"\n**Latest:** {last_val} ({last_date}) | "
            f"**Change from {first_date} to {last_date}:** {delta:+.2f}{pct}, "
            f"from {first_val}\n"
        )
    except ValueError:
        summary = f"\n**Latest:** {last_val} ({last_date})\n"

    shown = points
    note = ""
    if len(points) > MAX_ROWS:
        shown = points[-MAX_ROWS:]
        note = f"\n_(showing the most recent {MAX_ROWS} of {len(points)} observations)_\n"

    table = (
        "\n| Date | Value |\n| --- | --- |\n"
        + "\n".join(f"| {d} | {v} |" for d, v in shown)
        + "\n"
    )

    return header + summary + note + table
