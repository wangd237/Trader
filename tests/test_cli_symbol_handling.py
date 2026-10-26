"""CLI 的代码校验和资产分类必须与数据层保持一致。

回归验证 #980（GC=F 被拒绝）、#981（BTCUSD 被误判为股票）、
#982（BTC-USDT 在 Yahoo 无法直接查询）。"""
import pytest

import cli.run as cli_run
from cli.models import AssetType
from cli.prompts import detect_asset_type, is_valid_ticker_input, normalize_ticker_symbol
from tradingagents.dataflows.symbols import normalize_symbol


# --- #982：稳定币计价加密资产转换为 Yahoo 的 -USD 交易对 ---
@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", "BTC-USD"),
    ("BTCUSDT", "BTC-USD"),
    ("BTC-USDT", "BTC-USD"),
    ("BTC-USDC", "BTC-USD"),
    ("ethusdt", "ETH-USD"),
    # 非加密资产代码保持原样。
    ("AAPL", "AAPL"),
    ("GC=F", "GC=F"),
    ("600519.SS", "600519.SS"),
    ("EURUSD", "EURUSD=X"),
])
def test_normalize_symbol_crypto_and_passthrough(raw, expected):
    assert normalize_symbol(raw) == expected


# --- #980：允许 Yahoo 期货/外汇代码 ---
@pytest.mark.parametrize("value,ok", [
    ("GC=F", True),
    ("EURUSD=X", True),
    ("AAPL", True),
    ("0700.HK", True),
    ("^GSPC", True),
    ("", True),                 # empty -> defaults to SPY downstream
    ("bad symbol!", False),     # space + '!' rejected
    ("A" * 40, False),          # too long
])
def test_ticker_input_validation(value, ok):
    assert is_valid_ticker_input(value) is ok


# --- #981/#982：基于标准化代码判断资产类型 ---
@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", AssetType.CRYPTO),
    ("BTC-USDT", AssetType.CRYPTO),
    ("BTC-USD", AssetType.CRYPTO),
    ("ETHUSD", AssetType.CRYPTO),
    ("AAPL", AssetType.STOCK),
    ("GC=F", AssetType.STOCK),
    ("600519.SS", AssetType.STOCK),
])
def test_detect_asset_type(raw, expected):
    assert detect_asset_type(raw) == expected


def test_cli_normalize_delegates_to_data_layer():
    # CLI 生成的标准化代码必须与数据层实际查询的代码相同。
    for raw in ("XAUUSD", "BTCUSD", "btc-usdt", "AAPL"):
        assert normalize_ticker_symbol(raw) == normalize_symbol(raw)


@pytest.mark.unit
def test_the_run_directory_cannot_escape_the_results_directory(tmp_path, monkeypatch):
    """其他使用 ticker 拼接路径的入口都有安全校验，
    CLI 的结果目录此前遗漏，输入 '..' 会写到上级目录。"""

    with pytest.raises(ValueError):
        cli_run._run_directory({"results_dir": str(tmp_path)}, "..", "2026-09-01")

    ok = cli_run._run_directory({"results_dir": str(tmp_path)}, "NVDA", "2026-09-01")
    assert str(ok).startswith(str(tmp_path))
