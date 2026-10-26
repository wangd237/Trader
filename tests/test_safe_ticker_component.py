"""测试防止目录遍历的 ticker 路径组件校验器。"""

import os
import unittest

import pytest

from tradingagents.dataflows.symbols import safe_ticker_component


@pytest.mark.unit
class TestSafeTickerComponent(unittest.TestCase):
    def test_accepts_common_ticker_formats(self):
        for ticker in ("AAPL", "BRK-B", "BRK.A", "0700.HK", "7203.T", "BHP.AX", "^GSPC"):
            self.assertEqual(safe_ticker_component(ticker), ticker)

    def test_accepts_futures_and_forex_formats(self):
        # 期货符号使用 =（例如黄金 GC=F、原油 CL=F），外汇和 CFD 代码可能包含 +。
        for ticker in ("GC=F", "CL=F", "ES=F", "XAUUSD+", "EURUSD+"):
            self.assertEqual(safe_ticker_component(ticker), ticker)

    def test_rejects_path_separators(self):
        for bad in (".", "..", "../etc", "a/b", "a\\b", "/abs", "..\\..\\x"):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_rejects_null_byte_and_whitespace(self):
        for bad in ("AAP L", "AAPL\x00", "AAPL\n", "\tAAPL"):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_rejects_empty_or_non_string(self):
        for bad in ("", None, 123, b"AAPL"):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_rejects_overlong_input(self):
        with self.assertRaises(ValueError):
            safe_ticker_component("A" * 33)

    def test_rejects_dot_only_values(self):
        # '.'、'..' 虽可通过字符正则检查，作为路径组件时仍可能指向其他目录，
        # 例如 ``Path(results_dir) / ticker / "logs"``。
        for bad in (".", "..", "...", "...."):
            with self.assertRaises(ValueError):
                safe_ticker_component(bad)

    def test_traversal_string_does_not_escape_join(self):
        """基本校验：安全的 ticker 拼接后仍位于指定根目录内部。"""
        base = os.path.realpath("/tmp/cache")
        ticker = safe_ticker_component("AAPL")
        joined = os.path.realpath(os.path.join(base, f"{ticker}.csv"))
        self.assertTrue(joined.startswith(base + os.sep))


if __name__ == "__main__":
    unittest.main()
