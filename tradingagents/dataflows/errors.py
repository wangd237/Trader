"""数据提供商的异常分类体系。

采用统一的异常继承结构，使路由层根据错误行为而不是提供商名称处理异常。
所有无法返回可用数据的情况都派生自 ``VendorError``，路由器捕获基类。
新增提供商只需抛出对应异常或薄封装子类，无需新增 except 分支。

VendorError
├── NoMarketDataError：没有可用记录（空结果或数据过期）
├── VendorUnavailableError：限流或调用失败，应尝试下一提供商
└── VendorNotConfiguredError：缺少 API Key 或配置，视为提供商不可用

异常类型数量取决于路由器需要做出多少种处理，而非文字上可区分的
错误原因。空数据和过期数据处理方式相同，因此共用
``NoMarketDataError``，仅通过 ``detail`` 描述具体差异。"""

from __future__ import annotations


class VendorError(Exception):
    """所有数据提供商无法返回可用数据时使用的异常基类。"""


class NoMarketDataError(VendorError):
    """数据提供商未能返回指定标的的可用记录（结果为空或数据过期）。

    同时保留用户请求的原始 ticker、实际查询的规范化 ticker，
    以及自由文本 ``detail``，方便调用者生成清晰的错误消息，
    避免将提供商专属的空字符串直接传给下游。"""

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class VendorUnavailableError(VendorError):
    """数据提供商因限流或请求失败而无法响应。

    这并不表示标的没有数据；路由器应继续尝试下一提供商。"""


class VendorNotConfiguredError(VendorError, ValueError):
    """已选择数据提供商，但缺少 API Key 或必需配置。

    该异常也继承 ``ValueError``，以兼容现有捕获逻辑；
    路由层则将其视为“提供商不可用”。"""
