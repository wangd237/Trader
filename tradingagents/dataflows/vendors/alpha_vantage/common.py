import json
import os
from datetime import datetime
from io import StringIO

import pandas as pd

from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorUnavailableError
from tradingagents.dataflows.net import get_scrubbed

API_BASE_URL = "https://www.alphavantage.co/query"

# 网络超时秒数，防止 Alpha Vantage 请求卡住导致 CLI 或 Agent
# 无限期等待（#990）。
REQUEST_TIMEOUT = 30


class AlphaVantageNotConfiguredError(VendorNotConfiguredError):
    """选择 Alpha Vantage 但未配置 API Key 时抛出的异常。

    它继承 VendorNotConfiguredError，也仍是 ValueError，
    因此路由器的“提供商不可用”分支和旧版捕获 ValueError 的调用方均可兼容。"""
    pass


def get_api_key() -> str:
    """从环境变量中读取 Alpha Vantage 的 API Key。"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise AlphaVantageNotConfiguredError(
            "ALPHA_VANTAGE_API_KEY environment variable is not set."
        )
    return api_key


def format_datetime_for_api(date_input, end_of_day: bool = False) -> str:
    """将不同日期格式转换成 Alpha Vantage 要求的 YYYYMMDDTHHMM。

    单独的日期表示当天零点，即当天开始；窗口结束时应传入
    ``end_of_day``，以包含分析日当天发布的所有数据。"""
    if isinstance(date_input, str):
        # 如果格式已经符合要求，则原样返回。
        if len(date_input) == 13 and 'T' in date_input:
            return date_input
        # 尝试解析常见日期格式。
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return dt.strftime("%Y%m%dT2359" if end_of_day else "%Y%m%dT0000")
        except ValueError:
            try:
                dt = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
                return dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                raise ValueError(f"Unsupported date format: {date_input}") from None
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")


class AlphaVantageRateLimitError(VendorUnavailableError):
    """Alpha Vantage API 超过速率限制时抛出的异常。"""
    pass


def _make_api_request(function_name: str, params: dict) -> dict | str:
    """发送 API 请求并处理响应的辅助函数。

    异常：
        AlphaVantageRateLimitError：API 请求触发限流时抛出。"""
    # 复制 params，避免修改调用方传入的原始参数。
    api_key = get_api_key()
    api_params = params.copy()
    api_params.update({
        "function": function_name,
        "apikey": api_key,
        "source": "trading_agents",
    })

    # 如果请求参数或全局配置包含 entitlement，进行对应处理。
    current_entitlement = globals().get('_current_entitlement')
    entitlement = api_params.get("entitlement") or current_entitlement

    if entitlement:
        api_params["entitlement"] = entitlement
    elif "entitlement" in api_params:
        # entitlement 为空或 None 时将其移除。
        api_params.pop("entitlement", None)

    response = get_scrubbed(
        API_BASE_URL, params=api_params, timeout=REQUEST_TIMEOUT, secret=api_key
    )

    response_text = response.text

    # 错误响应通常是 JSON，正常数据通常是 CSV 或以字段组织的 JSON；
    # 非 JSON 响应不能直接视为异常。
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    # Alpha Vantage 通过 Information 或 Note 字段报告异常。
    # 必须区分真正的限流与无效/缺失密钥（#991）。
    # 限流提示也可能提及 API Key，因此应先检查限流特征。
    def scrubbed(text: str) -> str:
        # Alpha Vantage 的提示文本可能回显密钥；
        # 错误文本会进入日志，必须避免凭据泄漏。
        return text.replace(api_key, "***") if api_key else text

    notice = response_json.get("Information") or response_json.get("Note")
    if notice:
        notice = scrubbed(notice)
        low = notice.lower()
        if any(m in low for m in ("rate limit", "requests per day", "call frequency", "premium")):
            raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit exceeded: {notice}")
        if "api key" in low or "apikey" in low:
            # 密钥无效时复用现有“未配置”异常，
            # 让用户看到可操作的鉴权错误，而不是误报为限流（#991）。
            raise AlphaVantageNotConfiguredError(f"Alpha Vantage API key invalid or missing: {notice}")

    # 接口拒绝请求时返回 {"Error Message": ...}，
    # 原因可能是未知代码，也可能是请求格式错误；
    # 不能仅凭它判断标的不存在（#1442）。
    rejection = response_json.get("Error Message")
    if rejection:
        raise VendorUnavailableError(f"Alpha Vantage rejected the request: {scrubbed(rejection)}")

    return response_text


def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    """按指定日期范围过滤 CSV 数据，仅保留范围内的记录。

    参数：
        csv_data：Alpha Vantage API 返回的 CSV 字符串。
        start_date：起始日期，yyyy-mm-dd。
        end_date：结束日期，yyyy-mm-dd。

    返回：
        筛选后的 CSV 字符串。"""
    if not csv_data or csv_data.strip() == "":
        return csv_data

    # TIME_SERIES_DAILY_ADJUSTED 默认返回直到今天的完整历史序列。
    # 因此这里按 end_date 截取是历史回测避免读取未来价格的关键防线。
    # 不能吞掉解析错误并返回未筛选的原始数据。
    df = pd.read_csv(StringIO(csv_data))

    # 假定第一列为日期（时间戳）列。
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered_df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

    return filtered_df.to_csv(index=False)
