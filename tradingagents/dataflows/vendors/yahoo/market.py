import logging
from datetime import datetime
from typing import Annotated

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from tradingagents.dataflows.errors import NoMarketDataError, VendorError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry
from tradingagents.dataflows.vendors.yahoo.ohlcv import _assert_ohlcv_not_stale, load_ohlcv

logger = logging.getLogger(__name__)


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # 将券商或外汇代码转换为 Yahoo 格式，例如 XAUUSD+ → GC=F。
    canonical = normalize_symbol(symbol)

    # yfinance 的 ``end`` 是不包含的右边界，会漏掉用户要求的
    # end_date 当天（若为今天则漏掉今日）的数据。
    # 因此向接口请求时把结束日期向后移动一天（#986/#987）。
    end_inclusive = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")
    data = yf_retry(lambda: yf.Ticker(canonical).history(start=start_date, end=end_inclusive))

    # 空结果可能表示代码未知或已经退市。抛出结构化异常而非返回文字，
    # 让路由层生成明确的无数据标记，避免 Agent 编造价格。
    if data is None or data.empty:
        raise_for_empty(symbol, canonical, f"rows between {start_date} and {end_date}")

    # 移除索引中的时区信息，简化输出。
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # 格式化报告前拒绝过期数据（如只返回一年前的部分记录）。
    # 抛出 NoMarketDataError，让路由器统一报告不可用（#1021）。
    _assert_ohlcv_not_stale(data, end_date, symbol, canonical)

    # 数值保留两位小数，便于展示。
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    csv_string = data.to_csv()

    # 规范化后的 ticker 与原输入不同，报告中应注明实际查询的标的。
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Stock data for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n\n"

    return header + csv_string


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    as_of_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # 移动平均线指标。
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD 相关指标。
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # 动量指标。
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # 波动率指标。
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # 成交量指标。
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = as_of_date
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    before = as_of_dt - relativedelta(days=look_back_days)

    # 优化：行情只下载一次，然后计算全部日期对应的指标。
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, as_of_date)

        # 生成需要计算的日期范围。
        current_dt = as_of_dt
        date_values = []

        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')

            # 查询当前日期对应的指标数值。
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"

            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"

    except VendorError:
        raise  # Unknown/delisted symbol — let the router emit the sentinel
    except Exception as e:
        logger.warning("Bulk stockstats fetch failed, falling back per-day: %s", e)
        # 批量计算失败时，回退到原始实现。
        ind_string = ""
        as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        while as_of_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, as_of_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{as_of_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            as_of_dt = as_of_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    as_of_date: Annotated[str, "current date for reference"]
) -> dict:
    """批量计算股票技术指标的优化实现。
    只获取一次历史行情，计算全部可用日期的指标，
    返回日期字符串到指标值的字典。"""
    from stockstats import wrap

    data = load_ohlcv(symbol, as_of_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df[indicator]  # This triggers stockstats to calculate the indicator

    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]

        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)

    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    as_of_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    as_of_date = as_of_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = get_stock_stats(
            symbol,
            indicator,
            as_of_date,
        )
    except VendorError:
        raise  # Unknown/delisted symbol — let the router emit the sentinel
    except Exception as e:
        # 空字符串会在指标表中显示成“2026-05-08: ”，
        # 这容易被理解为当天没有指标数据，而实际可能是读取失败。
        # 应抛出异常，让路由层尝试下一提供商或报告序列不可用。
        raise NoMarketDataError(
            symbol, symbol, f"{indicator} could not be read for {as_of_date}: {e}"
        ) from e

    return str(indicator_value)


def get_closes(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """获取从 ``start_date`` 开始、到 ``end_date`` 之前的每日收盘价。"""
    canonical = normalize_symbol(symbol)
    history = yf_retry(lambda: yf.Ticker(canonical).history(start=start_date, end=end_date))
    return history["Close"] if history is not None and "Close" in history else pd.Series(dtype=float)


def get_stock_stats(
    symbol: Annotated[str, "ticker symbol for the company"],
    indicator: Annotated[
        str, "quantitative indicators based off of the stock data for the company"
    ],
    as_of_date: Annotated[
        str, "curr date for retrieving stock price data, YYYY-mm-dd"
    ],
):
    data = load_ohlcv(symbol, as_of_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    as_of_str = pd.to_datetime(as_of_date).strftime("%Y-%m-%d")

    df[indicator]  # trigger stockstats to calculate the indicator
    matching_rows = df[df["Date"].str.startswith(as_of_str)]

    if not matching_rows.empty:
        indicator_value = matching_rows[indicator].values[0]
        return indicator_value
    else:
        return "N/A: Not a trading day (weekend or holiday)"
