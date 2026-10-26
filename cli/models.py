from enum import StrEnum


class AnalystType(StrEnum):
    MARKET = "market"
    # 序列化标识保持 "social"，兼容旧配置和字符串键调用者。
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class AssetType(StrEnum):
    STOCK = "stock"
    CRYPTO = "crypto"
