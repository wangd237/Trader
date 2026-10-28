"""Offline tests for the first A-share vendor integration."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.vendors import ashare


def _bars(volume):
    return pd.DataFrame([{
        "date": "2026-09-30", "open": 11.50, "high": 11.66,
        "low": 11.43, "close": 11.57, "volume": volume,
        "amount": 1_206_000_000,
    }])


def test_sz000_volume_conversion(monkeypatch):
    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: _bars(1_045_357)),
    )
    report = ashare.get_stock_data("000001.SZ", "2026-09-01", "2026-09-30")
    assert "104535700" in report
    assert "# Source: Tencent" in report


def test_volume_not_doubled_if_already_shares(monkeypatch):
    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: _bars(104_535_700)),
    )
    report = ashare.get_stock_data("000001.SZ", "2026-09-01", "2026-09-30")
    assert "104535700" in report
    assert "10453570000" not in report


def test_disclosure_as_of_filters_future_statement(monkeypatch):
    df = pd.DataFrame([
        {"报告日": "20260930", "公告日期": "20261021", "营业收入": 300},
        {"报告日": "20260630", "公告日期": "20260815", "营业收入": 200},
    ])
    monkeypatch.setattr(
        ashare, "_ak",
        lambda: SimpleNamespace(stock_financial_report_sina=lambda **kwargs: df),
    )
    report = ashare.get_income_statement("300750.SZ", "quarterly", "2026-09-30")
    assert "20260630" in report
    assert "20260930" not in report
    assert "20260815" in report


def test_invalid_symbol_rejected():
    with pytest.raises(ValueError, match="Not an A-share"):
        ashare.get_stock_data("AAPL", "2026-09-01", "2026-09-30")


def test_router_registers_ashare_vendor():
    from tradingagents.dataflows.router import VENDOR_METHODS
    for method in ("get_stock_data", "get_balance_sheet", "get_cashflow", "get_income_statement"):
        assert "ashare" in VENDOR_METHODS[method]
