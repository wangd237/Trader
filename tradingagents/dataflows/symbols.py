"""对数据源调用中的交易标的代码进行规范化，同时验证 ticker 是否可安全用作路径组件。

Yahoo Finance（默认数据源）使用的符号格式与券商、TradingView 或 MT5
常见格式不同，例如：

用户输入       Yahoo 代码       说明
XAUUSD         GC=F            黄金按 COMEX 期货表示
EURUSD         EURUSD=X        即期外汇使用 =X 后缀
BTCUSD         BTC-USD         加密资产使用连字符
SPX500         ^GSPC           指数 CFD 对应 Yahoo 指数
700.HK         0700.HK         港股代码补齐四位数字
600519.SH      600519.SS       上海交易所代码使用 .SS

直接将券商代码传给 Yahoo 可能得到空结果，导致 Agent 在缺少数据时
臆测价格（#781）。集中管理映射可保证全部 yfinance 入口行为一致；
新增标的只需扩展映射表，无须修改每个调用点。"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# 常见的 ISO-4217 外汇货币代码。若六字母代码的前后三位
# 均在列表中，则视为即期外汇交易对，并添加 Yahoo 的 ``=X`` 后缀。
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

# 券商可能直接拼接 USD 而不加分隔符的加密资产基础币种。
_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

# 无法用通用规则映射到 Yahoo 代码的显式别名表。
# 金属和能源映射到近期主力期货，指数 CFD 映射到 Yahoo 指数。
# 新增项目只需添加表项，无需更改调用代码。
_ALIASES = {
    # 贵金属：现货名称映射到 COMEX/NYMEX 期货。
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # 能源品种。
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # 指数 CFD 映射到 Yahoo 指数代码。
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# 港交所代码按 Yahoo 格式规范化：数字补零至四位（#957）。
_HK_CODE = re.compile(r"^(\d{1,5})\.HK$")
_SHANGHAI_SH = re.compile(r"^(\d{6})\.SH$")


# 加密资产计价货币均映射到 Yahoo 的美元交易对。
# Yahoo 只提供 ``<BASE>-USD``，不提供 USDT/USDC 稳定币交易对；
# 因此这些计价方式最终都转换为 ``-USD``（#982）。
# 按长度由长到短匹配，确保 USDT/USDC 优先于 USD 子串。
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")


def crypto_base(raw: str) -> str | None:
    """对已知以 USD、USDT 或 USDC 计价的加密资产代码，
    返回基础币种（如 ``BTC``）。支持 ``BTC-USD``、``BTCUSD``、
    ``BTC-USDT`` 等写法；非加密资产返回 None。
    仅做字符串规则匹配，不访问网络。"""
    if not isinstance(raw, str):
        return None
    compact = raw.strip().upper().rstrip("+").replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            return base if base in _CRYPTO_BASES else None
    return None


def _normalize_crypto(s: str) -> str | None:
    """将已知的 USD、USDT、USDC 计价加密资产规范化为 ``<BASE>-USD``；否则返回 None。"""
    base = crypto_base(s)
    return f"{base}-USD" if base else None


def normalize_symbol(raw: str) -> str:
    """将用户或券商使用的代码映射为 Yahoo Finance 标准 ticker。

    按以下优先级匹配（首先命中的规则生效）：
    1. 显式别名表：贵金属、能源和指数 CFD。
    2. 加密资产：已知基础币种以 USD、USDT 或 USDC 计价时，
       统一规范化为 ``BASE-USD``，允许带或不带连字符。
    3. 外汇：由两个 ISO 货币代码构成的六字母字符串映射为 ``PAIR=X``。
    4. 港股：数字 ``.HK`` 代码统一补齐四位，
       如 ``09992.HK`` → ``9992.HK``，``700.HK`` → ``0700.HK``。
    5. 沪市：``600519.SH`` → ``600519.SS``。
    6. 其他情况：原样保留符号内容并转为大写，适用于普通股票、ETF
       和 ``GC=F``、``^GSPC`` 等 Yahoo 原生代码。

    匹配前移除券商 CFD 的尾部 ``+`` 标记（如 ``XAUUSD+``）。
    本函数仅做语法转换，不调用网络，因此适合每次请求时执行。"""
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Yahoo 不使用的券商 CFD 或其他修饰后缀。
    s = s.rstrip("+")

    crypto = _normalize_crypto(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    elif hk := _HK_CODE.match(s):
        canonical = f"{int(hk.group(1)):04d}.HK"
    elif sh := _SHANGHAI_SH.match(s):
        canonical = f"{sh.group(1)}.SS"
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


# ticker 可包含字母、数字、点、连字符、下划线、插入符
# （如指数 ^GSPC）、等号（如期货 GC=F）和加号
# （如外汇或 CFD 代码 XAUUSD+）。这些字符不会导致路径向上遍历；
# 其余字符一律拒绝，防止拼接后的路径逃逸。
_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\^=+]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    """验证 ``value`` 是否可以安全拼接到文件系统路径。

    ticker 可能来自用户 CLI 输入或 LLM 工具调用，二者都可能受外部内容
    影响，例如新闻中的 Prompt 注入。如果不校验，
    ``"../../../etc/foo"`` 可能经 ``os.path.join`` 或 ``Path /``
    逃逸到缓存、检查点或结果目录之外。

    通过允许字符校验时原样返回 ``value``；否则抛出 ``ValueError``。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    # 上面的正则允许点号，因此单独的 "."、".."、"..." 也会匹配；
    # 它们作为路径组件可能指向父目录，需要单独拒绝全由点构成的值。
    if set(value) == {"."}:
        raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
    return value
