"""Yahoo 数据请求共用的重试逻辑，以及失败或空结果的统一处理。"""

import logging
import time

import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.net import vendor_reachable

logger = logging.getLogger(__name__)

YAHOO_HOST = "https://query2.finance.yahoo.com"


def raise_for_empty(symbol: str, canonical: str, what: str) -> None:
    """判断 Yahoo 的空响应代表没有数据还是服务故障。

    如果 Yahoo 本身无法访问，空响应就不能证明指定 ticker 不存在。
    因此先检查服务可达性，再报告该标的没有对应数据。"""
    if not vendor_reachable(YAHOO_HOST):
        raise VendorUnavailableError(f"Yahoo Finance is unreachable; no {what} was retrieved")
    raise NoMarketDataError(symbol, canonical, f"no {what}")


# yfinance 对部分失败请求返回空结果，容易被误当成“标的无数据”；
# 需要将这类请求故障明确作为异常报告。
yf.config.debug.hide_exceptions = False


def _answered_empty(exc: Exception) -> bool:
    """判断 Yahoo 是否确实回答“无数据”，例如代码不存在（HTTP 404），
    或某段价格窗口内没有行情。

    yfinance 的 YFPricesMissingError 既可能表示无价格，也可能包装了
    请求错误，因此只将明确的无数据响应视为不存在。
    缺失时区信息同样不能证明标的不存在，因为失败查询也会产生类似提示。"""
    if isinstance(exc, YFPricesMissingError):
        reason = exc.yahoo_reason
        return "status_code" not in (exc.debug_info or "") and (
            reason is None or reason.startswith("Data doesn't exist"))
    return getattr(getattr(exc, "response", None), "status_code", None) == 404


def yf_retry(func, max_retries=3, base_delay=2.0):
    """对 yfinance 调用实施指数退避重试。

    yfinance 遇到 HTTP 429 时抛出 YFRateLimitError，却不会自行重试。
    若重试耗尽或出现其他调用异常，则抛出 VendorUnavailableError，
    只表示提供商调用失败，不能推断标的情况。
    若 Yahoo 明确没有该标的数据，返回 None。
    ``func`` 应自行创建 Ticker 并请求，因为同一个 Ticker 会记住失败的
    ``info`` 请求，再次读取该实例可能直接返回空概况。"""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise VendorUnavailableError(
                    f"Yahoo Finance rate limited after {max_retries} retries: {exc}"
                ) from exc
        except Exception as exc:
            if _answered_empty(exc):
                return None
            raise VendorUnavailableError(f"Yahoo Finance request failed: {type(exc).__name__}") from exc
