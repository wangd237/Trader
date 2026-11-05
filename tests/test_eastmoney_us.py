"""No-network regression tests for Eastmoney BABA market history."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import eastmoney_us


@pytest.fixture
def mock_eastmoney(monkeypatch):
    eastmoney_us._historical_frame.cache_clear()
    calls = []
    dates = pd.bdate_range("2025-08-01", "2026-11-05")
    values = pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": 80.0 + pd.Series(range(len(dates))) * 0.1,
        "最高": 80.4 + pd.Series(range(len(dates))) * 0.1,
        "最低": 79.7 + pd.Series(range(len(dates))) * 0.1,
        "收盘": 80.2 + pd.Series(range(len(dates))) * 0.1,
        "成交量": 500_000 + pd.Series(range(len(dates))) * 100,
    })

    def download(**kwargs):
        calls.append(kwargs)
        return values.copy()

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_us_hist=download))
    yield calls
    eastmoney_us._historical_frame.cache_clear()


@pytest.mark.unit
def test_baba_price_history_has_correct_mapping_and_no_future_bars(mock_eastmoney):
    report = eastmoney_us.get_stock_data("BABA", "2026-11-02", "2026-11-04")
    assert "2026-11-05" not in report
    assert "2026-11-04" in report
    assert "Source: Eastmoney US" in report
    assert mock_eastmoney[0] == {
        "symbol": "106.BABA",
        "period": "daily",
        "start_date": "20261102",
        "end_date": "20261104",
        "adjust": "",
    }


@pytest.mark.unit
def test_reuse_historical_bars_across_indicators_and_snapshot(mock_eastmoney):
    first = eastmoney_us.get_indicators("BABA", "close_50_sma", "2026-11-04", 10)
    second = eastmoney_us.get_indicators("BABA", "rsi", "2026-11-04", 10)
    data = eastmoney_us.load_ohlcv("BABA", "2026-11-04")
    assert "2026-11-05" not in first
    assert "Source: Eastmoney" in second
    assert data["Date"].max() == pd.Timestamp("2026-11-04")
    assert len(mock_eastmoney) == 1, "indicator calls share the same historical bars"


@pytest.mark.unit
def test_optional_us_vendor_is_available_when_explicitly_selected(mock_eastmoney):
    from tradingagents.dataflows.config import run_config
    from tradingagents.dataflows.router import route_to_vendor

    with run_config({"data_vendors": {"core_stock_apis": "eastmoney_us"}}):
        out = route_to_vendor("get_stock_data", "BABA", "2026-11-02", "2026-11-04")
    assert "Source: Eastmoney US" in out


@pytest.mark.unit
def test_market_snapshot_uses_same_vendor(mock_eastmoney):
    from tradingagents.agents.tools import get_verified_market_snapshot

    from tradingagents.dataflows.config import run_config

    with run_config({"data_vendors": {"core_stock_apis": "eastmoney_us"}}):
        out = get_verified_market_snapshot.func("BABA", "2026-11-04", 5)
    assert "Source: Eastmoney US" in out
    assert "Latest trading row used: 2026-11-04" in out
    assert "2026-11-05" not in out


@pytest.mark.unit
def test_unknown_us_ticker_does_not_guess_exchange_code():
    with pytest.raises(NoMarketDataError, match="not mapped"):
        eastmoney_us.get_stock_data("NOTABABA", "2026-11-02", "2026-11-04")


@pytest.mark.unit
def test_missing_ohlcv_rows_are_not_reported_as_valid(monkeypatch, mock_eastmoney):
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_us_hist=lambda **kwargs: pd.DataFrame(),
    ))
    eastmoney_us._historical_frame.cache_clear()
    with pytest.raises(NoMarketDataError, match="no daily prices"):
        eastmoney_us.get_stock_data("BABA", "2026-11-02", "2026-11-04")
