"""测试 FRED 宏观提供商：别名解析、配置异常、输出格式、缺失值、
历史时间窗口及路由集成。所有请求均使用 mock，不需要网络或密钥。"""
import copy
import unittest
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.vendors import fred

# 构造少量稳定的观测值，用于检查格式化结果。
_META = {
    "seriess": [
        {
            "title": "Unemployment Rate",
            "units_short": "%",
            "frequency": "Monthly",
            "seasonal_adjustment_short": "SA",
        }
    ]
}
_OBS = {
    "observations": [
        {"date": "2025-06-01", "value": "4.1"},
        {"date": "2025-07-01", "value": "4.3"},
        {"date": "2025-08-01", "value": "."},   # missing -> skipped
        {"date": "2025-09-01", "value": "4.4"},
    ]
}


def _request_stub(meta=_META, obs=_OBS):
    """构造按端点路径分发的 _request 模拟实现。"""
    def _impl(path, params):
        if path == "series":
            return meta
        if path == "series/observations":
            return obs
        raise AssertionError(f"unexpected FRED path: {path}")
    return _impl


@pytest.mark.unit
class FredResolutionTests(unittest.TestCase):
    def test_alias_maps_to_series_id(self):
        self.assertEqual(fred._resolve_series_id("cpi"), "CPIAUCSL")
        self.assertEqual(fred._resolve_series_id("unemployment"), "UNRATE")

    def test_alias_is_case_and_separator_insensitive(self):
        self.assertEqual(fred._resolve_series_id("Fed Funds Rate"), "FEDFUNDS")
        self.assertEqual(fred._resolve_series_id("10y-treasury"), "DGS10")

    def test_euro_area_aliases_map_to_series_ids(self):
        # FRED 也包含欧元区宏观数据，如 ECB 利率、Eurostat HICP/GDP、
        # OECD 长期国债收益率，可供 .PA 等非美股标的使用。
        expected = {
            "ecb_deposit_rate": "ECBDFR",
            "ecb_main_refi_rate": "ECBMRRFR",
            "euro_hicp": "CP0000EZ19M086NEST",
            "euro_core_hicp": "00XEFDEZ19M086NEST",
            "euro_real_gdp": "CLVMNACSCAB1GQEA19",
            "germany_10y": "IRLTLT01DEM156N",
            "france_10y": "IRLTLT01FRM156N",
            "eur_usd": "DEXUSEU",
        }
        for alias, series_id in expected.items():
            with self.subTest(alias=alias):
                self.assertEqual(fred._resolve_series_id(alias), series_id)
        self.assertEqual(fred._resolve_series_id("ECB Deposit Rate"), "ECBDFR")
        self.assertEqual(fred._resolve_series_id("euro-hicp"), "CP0000EZ19M086NEST")

    def test_unknown_alias_is_treated_as_raw_series_id(self):
        # 允许高级用户直接填写 FRED ID，按约定统一转为大写。
        self.assertEqual(fred._resolve_series_id("dgs30"), "DGS30")
        self.assertEqual(fred._resolve_series_id("MyCustomSeries"), "MYCUSTOMSERIES")

    def test_descriptive_phrase_is_rejected(self):
        # LLM 输出的自然语言描述（含空格或太长）不属于合法序列 ID；
        # 应在发起请求前拒绝，并给出提示，而非等待 API 400。
        for bad in ("bank of japan rate", "the unemployment number", "X" * 31):
            with self.assertRaises(ValueError):
                fred._resolve_series_id(bad)

    def test_get_macro_data_returns_guidance_on_bad_indicator(self):
        # 无效指标应产生可操作的提示，不调用 API，更不能崩溃。
        out = fred.get_macro_data("bank of japan rate", "2026-01-01")
        self.assertIn("FRED", out)
        self.assertIn("not a known macro alias", out)


@pytest.mark.unit
class FredConfigTests(unittest.TestCase):
    def test_missing_key_raises_not_configured(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
                self.assertRaises(fred.FredNotConfiguredError):
            fred.get_api_key()

    def test_not_configured_is_a_value_error(self):
        # 路由器依靠异常继承关系进行“提供商不可用”处理。
        self.assertTrue(issubclass(fred.FredNotConfiguredError, ValueError))


@pytest.mark.unit
class FredFormattingTests(unittest.TestCase):
    def test_report_has_header_latest_change_and_table(self):
        with mock.patch.object(fred, "_request", side_effect=_request_stub()):
            out = fred.get_macro_data("unemployment", "2025-09-30", 365)
        self.assertIn("## FRED: Unemployment Rate (UNRATE)", out)
        self.assertIn("Units: %", out)
        self.assertIn("Frequency: Monthly (SA)", out)
        self.assertIn("**Latest:** 4.4 (2025-09-01)", out)
        # 变化量标签应反映实际覆盖的观测点，而非请求的回溯天数；
        # 月度数据只有三个月观测时，不能误称同比变化。
        self.assertIn("**Change from 2025-06-01 to 2025-09-01:** +0.30 (+7.32%), from 4.1", out)
        self.assertNotIn("Change over window", out)
        self.assertIn("| 2025-06-01 | 4.1 |", out)

    def test_missing_value_is_skipped(self):
        with mock.patch.object(fred, "_request", side_effect=_request_stub()):
            out = fred.get_macro_data("unemployment", "2025-09-30", 365)
        # 缺失值 "." 不能出现在有效观测记录中。
        self.assertNotIn("2025-08-01", out)

    def test_empty_window_reports_no_observations(self):
        empty = {"observations": []}
        with mock.patch.object(fred, "_request", side_effect=_request_stub(obs=empty)):
            out = fred.get_macro_data("unemployment", "2025-09-30", 30)
        self.assertIn("No observations", out)

    def test_unknown_series_returns_not_found_message(self):
        # 合法格式但未知的序列 ID 应返回提示，不使可选宏观数据故障终止运行。
        no_series = {"seriess": []}
        with mock.patch.object(fred, "_request", side_effect=_request_stub(meta=no_series)):
            out = fred.get_macro_data("totally_unknown_xyz", "2025-09-30", 30)
        self.assertIn("not found", out)

    def test_long_series_is_truncated_but_change_uses_full_range(self):
        # 构造超过 MAX_ROWS 的确定性观测记录。
        obs = {
            "observations": [
                {"date": f"2025-01-{(i % 28) + 1:02d}", "value": str(i)}
                for i in range(fred.MAX_ROWS + 10)
            ]
        }
        with mock.patch.object(fred, "_request", side_effect=_request_stub(obs=obs)):
            out = fred.get_macro_data("unemployment", "2025-12-31", 365)
        self.assertIn(f"most recent {fred.MAX_ROWS}", out)
        # 变化量必须使用窗口内真正的首值（0）和末值。
        self.assertIn("+49.00, from 0\n", out)
        body_rows = [ln for ln in out.splitlines() if ln.startswith("| 2025")]
        self.assertEqual(len(body_rows), fred.MAX_ROWS)

    def test_window_is_lookahead_safe(self):
        # 观测结束日期必须等于 as_of_date，避免历史查询获取未来数据。
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("unemployment", "2025-09-30", 90)
        obs_params = captured["series/observations"]
        self.assertEqual(obs_params["observation_end"], "2025-09-30")
        self.assertEqual(obs_params["observation_start"], "2025-07-02")  # 90d back

    def test_requests_pin_the_data_vintage(self):
        # #1275：元数据和观测数据请求都必须把历史版本固定到 as_of_date，
        # 并以 FRED 当地今天为上限；否则会读取最新修订值造成未来信息泄漏。
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_fred_today", return_value="2026-01-01"), \
                mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("cpi", "2025-09-30", 90)

        for path in ("series", "series/observations"):
            self.assertEqual(captured[path]["realtime_start"], "2025-09-30", path)
            self.assertEqual(captured[path]["realtime_end"], "2025-09-30", path)

    def test_future_curr_date_clamps_vintage_to_fred_today(self):
        # #1275 回归：实时运行采用调用方本地日期，
        # 可能比 FRED 美国中部时间早进入下一天。
        # 若请求未来版本日期会返回 400，导致宏观数据被路由层静默丢弃。
        # 因此历史版本日期须限制为 FRED 今天，
        # 但观测窗口仍使用原 as_of_date。
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_fred_today", return_value="2026-08-31"), \
                mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("cpi", "2026-09-01", 90)  # local a day ahead of Chicago

        for path in ("series", "series/observations"):
            self.assertEqual(captured[path]["realtime_start"], "2026-08-31", path)
            self.assertEqual(captured[path]["realtime_end"], "2026-08-31", path)
        # 观测窗口仍跟随 as_of_date，而不是截断后的版本日期。
        self.assertEqual(captured["series/observations"]["observation_end"], "2026-09-01")


@pytest.mark.unit
class FredRoutingTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_macro_category_routes_to_fred(self):
        self.assertEqual(
            router.get_category_for_method("get_macro_indicators"), "macro_data"
        )
        set_config({"data_vendors": {"macro_data": "fred"}})
        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_macro_indicators": {"fred": lambda *a, **k: "MACRO_OK"}},
            clear=False,
        ):
            out = router.route_to_vendor("get_macro_indicators", "cpi", "2026-06-01", 365)
        self.assertEqual(out, "MACRO_OK")

    def test_not_configured_degrades_gracefully(self):
        # macro_data 是可选数据；仅配置 FRED 又缺少密钥时，
        # 路由层应返回不可用标记，不应中止整个分析。
        set_config({"data_vendors": {"macro_data": "fred"}})

        def _unconfigured(*a, **k):
            raise fred.FredNotConfiguredError("FRED_API_KEY not set")

        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_macro_indicators": {"fred": _unconfigured}},
            clear=False,
        ):
            out = router.route_to_vendor("get_macro_indicators", "cpi", "2026-06-01", 365)
        self.assertIn("DATA_UNAVAILABLE", out)


if __name__ == "__main__":
    unittest.main()


_KEY = "abcdef0123456789abcdef0123456789"


@pytest.mark.unit
class TestKeyKeptOutOfErrors:
    """密钥位于 URL 查询参数中，requests 可能把完整 URL 写入异常，
    导致日志或堆栈泄漏凭据（#1324）。"""

    def _raises(self, side_effect):
        with mock.patch.dict("os.environ", {"FRED_API_KEY": _KEY}), \
             mock.patch("tradingagents.dataflows.net.requests.get", side_effect=side_effect), \
             pytest.raises(requests.RequestException) as caught:
            fred._request("series", {"series_id": "DGS10"})
        return caught.value

    def test_http_error_message_carries_no_key(self):
        response = mock.Mock(status_code=502)
        response.raise_for_status.side_effect = requests.HTTPError(
            f"502 Server Error for url: https://api.stlouisfed.org/fred/series?api_key={_KEY}",
            response=response,
        )
        with mock.patch.dict("os.environ", {"FRED_API_KEY": _KEY}), \
             mock.patch("tradingagents.dataflows.net.requests.get", return_value=response), \
             pytest.raises(requests.HTTPError) as caught:
            fred._request("series", {"series_id": "DGS10"})
        exc = caught.value
        assert _KEY not in str(exc) and _KEY not in repr(exc)
        # 请求和响应中保存完整 URL，异常里不能继续附带这些对象。
        assert exc.response is None and exc.request is None
        assert exc.__cause__ is None and exc.__context__ is None  # no chain holds the key

    def test_connection_error_before_any_response_carries_no_key(self):
        exc = self._raises(requests.ConnectionError(
            f"Max retries exceeded with url: /fred/series?series_id=DGS10&api_key={_KEY}"))
        assert isinstance(exc, requests.ConnectionError)
        assert _KEY not in str(exc) and exc.__context__ is None


@pytest.mark.unit
def test_error_without_the_key_in_its_message_still_drops_the_request():
    # 部分超时异常文本不含 URL，但绑定的 request 仍可能携带密钥。
    import requests as rq
    req = rq.Request("GET", f"https://api.stlouisfed.org/fred/series?api_key={_KEY}").prepare()
    with mock.patch.dict("os.environ", {"FRED_API_KEY": _KEY}), \
         mock.patch("tradingagents.dataflows.net.requests.get", side_effect=rq.Timeout("Read timed out.", request=req)), \
         pytest.raises(rq.Timeout) as caught:
        fred._request("series", {"series_id": "DGS10"})
    assert caught.value.request is None
