import logging

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.vendors import ashare
from tradingagents.dataflows.vendors.alpha_vantage import (
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_global_news as get_alpha_vantage_global_news,
    get_income_statement as get_alpha_vantage_income_statement,
    get_indicator as get_alpha_vantage_indicator,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_stock as get_alpha_vantage_stock,
)
from tradingagents.dataflows.vendors.fred import get_macro_data as get_fred_macro_data
from tradingagents.dataflows.vendors.polymarket import (
    get_prediction_markets as get_polymarket_prediction_markets,
)
from tradingagents.dataflows.vendors.sec_edgar import (
    get_balance_sheet as get_sec_edgar_balance_sheet,
    get_cashflow as get_sec_edgar_cashflow,
    get_income_statement as get_sec_edgar_income_statement,
)
from tradingagents.dataflows.vendors.yahoo.fundamentals import (
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_fundamentals as get_yfinance_fundamentals,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from tradingagents.dataflows.vendors.yahoo.market import (
    get_stock_stats_indicators_window,
    get_YFin_data_online,
)
from tradingagents.dataflows.vendors.yahoo.news import get_global_news_yfinance, get_news_yfinance

logger = logging.getLogger(__name__)

# 按类别组织的工具。
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    },
    "macro_data": {
        "description": "Macroeconomic indicators (rates, inflation, labor, growth)",
        "tools": [
            "get_macro_indicators",
        ]
    },
    "prediction_markets": {
        "description": "Market-implied probabilities for forward-looking events",
        "tools": [
            "get_prediction_markets",
        ]
    }
}

# 可选的数据增强类别，为新闻分析师补充宏观和事件背景，
# 但不是生成决策的核心依赖。因此即使提供商失败，也只返回缺失标记，
# 不应让分析中止：例如 LLM 提供了错误指标、缺少密钥或网络短时异常。
# 价格、基本面和新闻等核心数据类别仍应抛出异常，明确暴露主数据源故障。
OPTIONAL_CATEGORIES = {"macro_data", "prediction_markets"}

# 方法与各数据提供商实现之间的映射。
VENDOR_METHODS = {
    # 股票核心数据接口。
    "get_stock_data": {
        "ashare": ashare.get_stock_data,
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    # 技术指标接口。
    "get_indicators": {
        "ashare": ashare.get_indicators,
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # 基本面数据接口。
    "get_fundamentals": {
        "ashare": ashare.get_fundamentals,
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "ashare": ashare.get_balance_sheet,
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "sec_edgar": get_sec_edgar_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "ashare": ashare.get_cashflow,
        "alpha_vantage": get_alpha_vantage_cashflow,
        "sec_edgar": get_sec_edgar_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "ashare": ashare.get_income_statement,
        "alpha_vantage": get_alpha_vantage_income_statement,
        "sec_edgar": get_sec_edgar_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # 新闻数据接口。
    "get_news": {
        "ashare": ashare.get_news,
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "ashare": ashare.get_insider_transactions,
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
    # 宏观数据接口。
    "get_macro_indicators": {
        "fred": get_fred_macro_data,
    },
    # 预测市场接口。
    "get_prediction_markets": {
        "polymarket": get_polymarket_prediction_markets,
    },
}


def get_category_for_method(method: str) -> str:
    """获取指定方法所属的数据类别。"""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str = None) -> str:
    """获取数据类别或某个工具方法配置的数据提供商。
    工具级配置优先于类别级配置。"""
    config = get_config()

    # 若指定了方法，先检查工具级配置。
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # 工具级配置不存在时，回退到类别级配置。
    return config.get("data_vendors", {}).get(category, "default")


def vendor_unavailable(method: str, error: Exception) -> str:
    """当所有提供商均被限流或无法访问时，构造调用返回结果。"""
    return (
        f"DATA_UNAVAILABLE: no configured vendor could serve {method} right now "
        f"({error}). This says nothing about the instrument; report the "
        f"data as unavailable and do not estimate or fabricate values."
    )


def no_data_available(error: NoMarketDataError) -> str:
    """当所有成功响应的提供商均无可用数据时，构造调用返回结果。"""
    resolved = "" if error.canonical == error.symbol else f" (resolved to '{error.canonical}')"
    # 保留具体类型异常的详细信息（如“最新一行日期为 2025-06-11，数据已过期”），
    # 便于 Agent 区分无效 ticker、无覆盖范围、数据过期等具体原因，
    # 而不仅看到笼统的“不可用”。
    reason = f" ({error.detail})" if error.detail else ""
    return (
        f"NO_DATA_AVAILABLE: No usable market data for '{error.symbol}'{resolved} from "
        f"any configured vendor{reason}. The symbol may be invalid, delisted, "
        f"not covered, or the vendor returned stale data. Do not estimate or "
        f"fabricate values — report that data is unavailable for this symbol."
    )


def route_to_vendor(method: str, *args, **kwargs):
    """按配置将方法调用路由到合适的数据提供商实现，并支持顺序回退。"""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    all_available_vendors = list(VENDOR_METHODS[method].keys())

    # Market-specific override at the data boundary; no Agent or graph changes.
    # Explicit per-tool vendor configuration remains authoritative.
    config = get_config()
    first_arg = args[0] if args else None
    is_ashare_request = (
        method in {"get_stock_data", "get_indicators", "get_fundamentals",
                   "get_balance_sheet", "get_cashflow", "get_income_statement",
                   "get_news", "get_insider_transactions"}
        and ashare.is_ashare(first_arg)
        and "ashare" in VENDOR_METHODS[method]
        and method not in config.get("tool_vendors", {})
    )
    if is_ashare_request:
        # Keep existing vendor fallback and DATA_UNAVAILABLE semantics.
        primary_vendors = ["ashare"]

    # 配置的数据提供商列表就是完整回退链，不会自动转向用户没有选择的
    # 提供商（#988/#289）。否则可能意外使用其他来源的数据，
    # 造成不同提供商之间的口径不一致。需要多数据源回退时按顺序写入，
    # 例如 data_vendors="yfinance,alpha_vantage"；无显式配置的
    # "default" 标记才表示使用全部可用提供商。
    explicit = [v for v in primary_vendors if v and v != "default"]
    if explicit:
        vendor_chain = [v for v in explicit if v in VENDOR_METHODS[method]]
        if not vendor_chain:
            raise ValueError(
                f"Configured vendor(s) {explicit} not available for '{method}'. "
                f"Available: {all_available_vendors}."
            )
    else:
        vendor_chain = all_available_vendors

    last_no_data: NoMarketDataError | None = None
    last_unavailable: VendorUnavailableError | None = None
    failed: Exception | None = None     # a vendor that raised something untyped
    first_error: Exception | None = None
    for vendor in vendor_chain:
        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except VendorUnavailableError as e:
            logger.warning("Vendor %r unavailable for %s: %s; trying next vendor.", vendor, method, e)
            # 保留错误信息，使所有提供商都不可用时能够明确指出
            # 是数据提供商故障，而不是误报标的没有数据。
            last_unavailable = e
            continue
        except VendorNotConfiguredError as e:
            logger.warning("Vendor %r not configured for %s; trying next vendor.", vendor, method)
            if first_error is None:
                first_error = e  # Surface it if no other vendor can serve the call.
            continue
        except NoMarketDataError as e:
            last_no_data = e  # No data here; another configured vendor may have it
            continue
        except Exception as e:
            # 如果其他提供商仍能提供数据，就不应让单个提供商失败导致调用中断；
            # 但也不能静默吞掉错误：主提供商故障必须写入日志（#989），
            # 不能被回退结果掩盖。
            logger.warning("Vendor %r failed for %s: %s", vendor, method, e)
            if first_error is None:
                first_error = e
            failed = e
            continue

    # 被限流或请求失败的提供商并没有确认是否持有该标的数据，
    # 因此不能依据其他提供商的“无数据”推断整条链均无数据。
    # 应报告提供商故障而非标的故障，同时不能直接终止分析。
    if last_unavailable is not None:
        return vendor_unavailable(method, last_unavailable)
    if failed is not None and last_no_data is not None:
        return vendor_unavailable(method, failed)

    # 所有成功响应的提供商都报告“无数据”，可以认为该标的不可用。
    # 返回统一且明确的缺失标记，而不是某家提供商特有的空字符串，
    # 从而让 Agent 报告“不可用”而不是编造数值。
    # 这一判断优先于回退过程中出现的次要错误。
    if last_no_data is not None:
        if first_error is not None:
            # 若其他提供商还出现过真实错误，应记录到日志，
            # 避免“无数据”结论掩盖主数据源的网络或鉴权故障。
            logger.warning(
                "Returning NO_DATA for %s, but a vendor errored earlier: %s",
                method, first_error,
            )
        return no_data_available(last_no_data)

    # 所有提供商都没有返回数据，也没有明确返回“无数据”时，
    # 应暴露第一个真实错误（例如主提供商网络故障）。
    # 可选增强类别则退化为缺失标记，避免非核心数据问题终止运行。
    if first_error is not None:
        if category in OPTIONAL_CATEGORIES:
            logger.warning("Optional %s unavailable for %s: %s", category, method, first_error)
            return (
                f"DATA_UNAVAILABLE: optional {category} could not be retrieved "
                f"({first_error}). Proceed without it; do not fabricate values."
            )
        raise first_error

    raise RuntimeError(f"No available vendor for '{method}'")
