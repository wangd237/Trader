"""Offline regression tests for domestic Hong Kong OHLCV and tool routing."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import hk


@pytest.fixture
def mock_hk_history(monkeypatch):
    hk._historic_download.cache_clear()
    dates = pd.bdate_range("2025-01-02", "2026-11-05")
    bars = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": 80 + pd.Series(range(len(dates))) / 10,
        "high": 81 + pd.Series(range(len(dates))) / 10,
        "low": 79 + pd.Series(range(len(dates))) / 10,
        "close": 80.5 + pd.Series(range(len(dates))) / 10,
        "volume": 400_000 + pd.Series(range(len(dates))) * 100,
    })
    calls = []

    def stock_hk_daily(**kwargs):
        calls.append(kwargs)
        return bars.copy()

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_hk_daily=stock_hk_daily
    ))
    yield calls
    hk._historic_download.cache_clear()


@pytest.mark.unit
def test_hk_code_normalisation_and_date_filter(mock_hk_history):
    report = hk.get_stock_data("9988.HK", "2026-11-02", "2026-11-04")
    assert mock_hk_history == [{"symbol": "09988", "adjust": ""}]
    assert "2026-11-04" in report
    assert "2026-11-05" not in report
    assert "unadjusted prices in HKD" in report
    assert hk._code("700.HK") == "00700"
    assert hk._code("0700.HK") == "00700"
    assert not hk.is_hk("BABA")


@pytest.mark.unit
def test_hk_indicator_and_snapshot_share_source_and_no_future(mock_hk_history):
    from tradingagents.agents.tools import get_verified_market_snapshot

    indicator = hk.get_indicators("9988.HK", "close_50_sma", "2026-11-04", 5)
    snapshot = get_verified_market_snapshot.func("9988.HK", "2026-11-04", 5)
    assert "Source: Sina Finance HK" in indicator
    assert "Latest trading row used: 2026-11-04" in snapshot
    assert "2026-11-05" not in indicator
    assert "2026-11-05" not in snapshot
    assert len(mock_hk_history) == 1


@pytest.mark.unit
def test_hk_price_routed_automatically_despite_us_default(mock_hk_history):
    from tradingagents.dataflows.config import run_config
    from tradingagents.dataflows.router import route_to_vendor

    with run_config({"data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
    }}):
        stock = route_to_vendor("get_stock_data", "9988.HK", "2026-11-02", "2026-11-04")
        indicator = route_to_vendor("get_indicators", "9988.HK", "rsi", "2026-11-04", 3)
    assert "Sina Finance HK" in stock
    assert "Sina Finance HK" in indicator
    assert len(mock_hk_history) == 1


@pytest.mark.unit
def test_explicit_provider_override_remains_authoritative(monkeypatch):
    from tradingagents.dataflows.config import run_config
    from tradingagents.dataflows.router import VENDOR_METHODS, route_to_vendor

    calls = []
    monkeypatch.setitem(VENDOR_METHODS["get_stock_data"], "yfinance",
                        lambda *args: calls.append(args) or "explicit")
    with run_config({"tool_vendors": {"get_stock_data": "yfinance"}}):
        result = route_to_vendor("get_stock_data", "9988.HK", "2026-11-02", "2026-11-04")
    assert result == "explicit"
    assert len(calls) == 1


@pytest.mark.unit
def test_stale_hk_bars_rejected(monkeypatch, mock_hk_history):
    hk._historic_download.cache_clear()
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_hk_daily=lambda **kwargs: pd.DataFrame([
            {"date": "2026-10-01", "open": 10, "high": 11,
             "low": 9, "close": 10, "volume": 1000}
        ])
    ))
    with pytest.raises(NoMarketDataError, match="stale"):
        hk.get_stock_data("9988.HK", "2026-09-01", "2026-11-04")


@pytest.mark.unit
def test_bad_hk_ticker_rejected():
    with pytest.raises(ValueError, match="Hong Kong"):
        hk.get_stock_data("BABA", "2026-11-01", "2026-11-04")



@pytest.mark.unit
def test_hk_parallel_cold_cache_miss_creates_only_one_native_decoder(
    monkeypatch, mock_hk_history,
):
    """LRU is internally synchronized but doesn't deduplicate in-flight misses."""
    downloader = sys.modules["akshare"].stock_hk_daily

    def slow_native_decoder(**kwargs):
        time.sleep(0.06)
        return downloader(**kwargs)

    monkeypatch.setattr(sys.modules["akshare"], "stock_hk_daily", slow_native_decoder)
    barrier = threading.Barrier(6)

    def fetch():
        barrier.wait(timeout=5)
        return hk.load_ohlcv("9988.HK", "2026-11-04")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: fetch(), range(6)))

    assert len(mock_hk_history) == 1, "native decoder must not see parallel cold-cache misses"
    assert all(len(frame) == len(results[0]) for frame in results)
    assert all(frame["Date"].max() == pd.Timestamp("2026-11-04") for frame in results)
