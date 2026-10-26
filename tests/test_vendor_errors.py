"""提供商错误类型体系：所有“无法返回可用数据”的异常都继承 VendorError，
因此路由器只需捕获基类，新数据源无需增加专门处理分支。"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.vendors.alpha_vantage.common import (
    AlphaVantageNotConfiguredError,
    AlphaVantageRateLimitError,
)
from tradingagents.dataflows.vendors.fred import FredNotConfiguredError


@pytest.mark.unit
class HierarchyTests(unittest.TestCase):
    def test_all_conditions_derive_from_vendor_error(self):
        for cls in (NoMarketDataError, VendorUnavailableError, VendorNotConfiguredError):
            self.assertTrue(issubclass(cls, VendorError))

    def test_not_configured_is_still_a_value_error(self):
        # 向后兼容：已有 except ValueError 的调用方应继续正常工作。
        self.assertTrue(issubclass(VendorNotConfiguredError, ValueError))

    def test_vendor_named_errors_subclass_the_generic_bases(self):
        self.assertTrue(issubclass(AlphaVantageRateLimitError, VendorUnavailableError))
        self.assertTrue(issubclass(AlphaVantageNotConfiguredError, VendorNotConfiguredError))
        self.assertTrue(issubclass(FredNotConfiguredError, VendorNotConfiguredError))
        # 因此这些异常仍属于 ValueError。
        self.assertTrue(issubclass(FredNotConfiguredError, ValueError))


@pytest.mark.unit
class RouterHandlesBaseTypesTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_rate_limit_subclass_caught_by_base(self):
        # 按提供商命名的限流异常应触发回退链尝试下一个数据源。
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})

        def _throttled(*a, **k):
            raise AlphaVantageRateLimitError("slow down")

        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _throttled, "yfinance": lambda *a, **k: "YF"}},
            clear=False,
        ):
            out = router.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(out, "YF")

    def test_not_configured_falls_through_to_next_vendor(self):
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})

        def _unconfigured(*a, **k):
            raise AlphaVantageNotConfiguredError("no key")

        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _unconfigured, "yfinance": lambda *a, **k: "YF"}},
            clear=False,
        ):
            out = router.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(out, "YF")

    def test_sole_unconfigured_vendor_surfaces_the_error(self):
        # 不存在后备提供商时，未配置错误必须明确暴露而不能被吞掉。
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})

        def _unconfigured(*a, **k):
            raise AlphaVantageNotConfiguredError("no key")

        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _unconfigured}},
            clear=False,
        ), self.assertRaises(AlphaVantageNotConfiguredError):
            router.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")

    def test_an_unavailable_vendor_keeps_absence_from_being_claimed(self):
        # 被限流的数据源尚未证明自己是否覆盖目标标的，
        # 因此不能断言“所有提供商都没有可用数据”。
        def throttled(*a, **k):
            raise VendorUnavailableError("Yahoo HTTP 429")

        def timed_out(*a, **k):
            raise TimeoutError("read timed out")    # a vendor that does not type its failures

        def not_covered(*a, **k):
            raise NoMarketDataError("AAPL", "AAPL", "not covered")

        for chain in ({"yfinance": throttled, "alpha_vantage": not_covered},
                      {"yfinance": not_covered, "alpha_vantage": throttled},
                      {"alpha_vantage": timed_out, "yfinance": not_covered}):
            set_config({"data_vendors": {"core_stock_apis": ",".join(chain)}})
            with mock.patch.dict(router.VENDOR_METHODS, {"get_stock_data": chain}, clear=False):
                out = router.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
            self.assertTrue(out.startswith("DATA_UNAVAILABLE"), out)


if __name__ == "__main__":
    unittest.main()
