
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import withhold_undisclosed_trades
from tradingagents.dataflows.vendors.alpha_vantage.common import (
    _make_api_request,
    format_datetime_for_api,
)


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """获取全球主要新闻媒体的实时及历史市场新闻与情绪数据。

    覆盖股票、加密资产、外汇，以及财政政策、并购和 IPO 等主题。

    参数：
        ticker：相关新闻的股票代码。
        start_date：新闻检索起始日期。
        end_date：新闻检索结束日期。

    返回：
        包含新闻情绪数据的字典或 JSON 字符串。"""

    # 未设置 limit 时接口默认返回 50 篇文章，
    # 每篇附带按 ticker 划分的情绪数据，最终全部进入 Prompt。
    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date, end_of_day=True),
        "limit": str(get_config()["news_article_limit"]),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def get_global_news(as_of_date, look_back_days: int | None = None, limit: int | None = None) -> dict[str, str] | str:
    """获取不按特定 ticker 筛选的全球市场新闻与情绪数据。

    覆盖金融市场、宏观经济等广泛主题。

    参数：
        as_of_date：分析日期，格式 yyyy-mm-dd。
        look_back_days：回溯天数；None 时读取当前配置的
            ``global_news_lookback_days``。
        limit：最大文章数量；None 时读取当前配置的
            ``global_news_article_limit``。

    返回：
        包含全球新闻情绪数据的字典或 JSON 字符串。"""
    from datetime import datetime, timedelta

    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(as_of_date, end_of_day=True),
        "limit": str(limit),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def get_insider_transactions(symbol: str, as_of_date: str | None = None) -> dict[str, str] | str:
    """获取关键利益相关者的最新和历史内部人交易记录。

    涵盖创始人、高管、董事等人员的交易。

    参数：
        symbol：ticker，例如 "IBM"。
        as_of_date：分析日期（yyyy-mm-dd）；历史日期时不返回此数据。

    返回：
        包含内部人交易数据的字典或 JSON 字符串。"""

    withheld = withhold_undisclosed_trades(as_of_date, symbol)
    if withheld:
        return withheld
    return _make_api_request("INSIDER_TRANSACTIONS", {"symbol": symbol})
