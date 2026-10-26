"""测试数据源空结果不会导致下游编造数据。

覆盖两个修复：load_ohlcv 不应缓存空下载结果以免污染缓存，
而应抛出 NoMarketDataError；route_to_vendor 在所有提供商
耗尽后，将该异常转换成统一的 NO_DATA_AVAILABLE 标记。"""

import os
import unittest
from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.yahoo import common, ohlcv


@pytest.mark.unit
class TestLoadOhlcvNoPoison(unittest.TestCase):
    def setUp(self):
        self._tmp = os.path.join(os.path.dirname(__file__), "_tmp_cache")
        os.makedirs(self._tmp, exist_ok=True)
        set_config({"data_cache_dir": self._tmp})

    def tearDown(self):
        for f in os.listdir(self._tmp):
            os.remove(os.path.join(self._tmp, f))
        os.rmdir(self._tmp)

    def test_empty_download_raises_and_does_not_cache(self):
        empty = mock.Mock(history=mock.Mock(return_value=pd.DataFrame()))
        # Yahoo 正常响应，因此空下载表示当前标的无数据。
        reachable = mock.patch.object(common, "vendor_reachable", return_value=True)
        reachable.start()
        self.addCleanup(reachable.stop)
        with mock.patch.object(ohlcv.yf, "Ticker", return_value=empty), \
                self.assertRaises(NoMarketDataError):
            ohlcv.load_ohlcv("FAKE", "2026-01-01")
        # 此时不能写入缓存。
        self.assertEqual(os.listdir(self._tmp), [])

        # 再次调用必须重新请求，不能使用被污染的空缓存。
        with mock.patch.object(ohlcv.yf, "Ticker", return_value=empty) as dl2:
            with self.assertRaises(NoMarketDataError):
                ohlcv.load_ohlcv("FAKE", "2026-01-01")
            self.assertTrue(dl2.called)


@pytest.mark.unit
class TestRouteToVendorSentinel(unittest.TestCase):
    def test_no_data_from_all_vendors_returns_sentinel(self):
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, "GC=F", "no rows")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_no_data}
        with mock.patch.dict(
            router.VENDOR_METHODS, {"get_stock_data": patched}, clear=False
        ):
            result = router.route_to_vendor(
                "get_stock_data", "XAUUSD+", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("XAUUSD+", result)
        self.assertIn("GC=F", result)
        self.assertIn("Do not estimate", result)

    def test_unconfigured_fallback_does_not_mask_no_data(self):
        # 主提供商明确无数据、后备提供商仅因缺少密钥等原因不可用时，
        # 应返回无数据标记，而非让后备提供商的次要异常终止调用。
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, symbol, "no rows")

        def raises_unavailable(symbol, *a, **k):
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_unavailable}
        with mock.patch.dict(
            router.VENDOR_METHODS, {"get_stock_data": patched}, clear=False
        ):
            result = router.route_to_vendor(
                "get_stock_data", "FAKE", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.unit
def test_an_unreachable_yahoo_is_not_reported_as_a_symbol_without_insider_data():
    from tradingagents.dataflows.errors import VendorUnavailableError
    from tradingagents.dataflows.vendors.yahoo import fundamentals

    ticker = type("T", (), {"insider_transactions": pd.DataFrame()})()
    with mock.patch.object(fundamentals.yf, "Ticker", return_value=ticker), \
         mock.patch.object(fundamentals, "vendor_reachable", return_value=False), \
         pytest.raises(VendorUnavailableError):
        fundamentals.get_insider_transactions("AAPL", as_of_date=get_current_date())
