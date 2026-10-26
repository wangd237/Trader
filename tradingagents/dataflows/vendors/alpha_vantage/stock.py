from datetime import datetime

from tradingagents.dataflows.vendors.alpha_vantage.common import (
    _filter_csv_by_date_range,
    _make_api_request,
)


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """返回指定日期范围内的原始日频 OHLCV、复权收盘价及历史拆股、股息事件。

    参数：
        symbol：股票代码，例如 IBM。
        start_date：开始日期，格式 yyyy-mm-dd。
        end_date：结束日期，格式 yyyy-mm-dd。

    返回：
        CSV 字符串，包含筛选后的日频复权时间序列。"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()

    # 根据请求区间是否落在最近 100 天内选择 outputsize。
    # compact 仅返回最近 100 个数据点，所以需要判断 start_date 是否足够近。
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 100 else "full"

    params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", params)

    return _filter_csv_by_date_range(response, start_date, end_date)
