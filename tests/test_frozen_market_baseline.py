"""Frozen Market Analyst: synthetic CSV unit test ONLY, never labelled real prices."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from tests.test_graph_end_to_end import ScriptedModel, TRADE_DATE, _Client
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.router import VENDOR_METHODS, route_to_vendor
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evaluation.collector import collect_trajectory
from tradingagents.evaluation.frozen_market import fixture_sha256, load_frozen_ohlcv
from tradingagents.graph import trading_graph


@pytest.fixture
def frozen_config(tmp_path):
    # Synthetic, monotonically increasing data: only checks isolation and
    # calculation integrity; never ship it as actual NVIDIA historical prices.
    dates = pd.bdate_range(end=TRADE_DATE, periods=260)
    closing = [100.0 + i / 10 for i in range(len(dates))]
    frame = pd.DataFrame({
        "Date": dates,
        "Open": [p - 0.1 for p in closing],
        "High": [p + 1 for p in closing],
        "Low": [p - 1 for p in closing],
        "Close": closing,
        "Volume": [1_000_000 + i for i in range(len(dates))],
    })
    file = tmp_path / "nvda-SYNTHETIC-TEST-ONLY.csv"
    frame.to_csv(file, index=False, date_format="%Y-%m-%d")
    config = deepcopy(DEFAULT_CONFIG)
    config.update({
        "benchmark_ohlcv_csv": str(file),
        "benchmark_ohlcv_sha256": fixture_sha256(file),
        "benchmark_symbol": "NVDA",
        "benchmark_company_name": "NVIDIA Corporation",
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
        "memory_log_path": str(tmp_path / "memory.md"),
        "checkpoint_enabled": False,
        "max_tool_rounds": 3,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    })
    return config


def _disable_vendor_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("NETWORK_VENDOR_WAS_CALLED")

    for methods in VENDOR_METHODS.values():
        for vendor in methods:
            monkeypatch.setitem(methods, vendor, forbidden)
    monkeypatch.setattr(trading_graph, "resolve_instrument_identity", forbidden)


@pytest.mark.unit
def test_frozen_market_routes_stock_indicators_snapshot_without_vendor(
    frozen_config, monkeypatch,
):
    _disable_vendor_network(monkeypatch)
    with run_config(frozen_config):
        prices = route_to_vendor(
            "get_stock_data", "NVDA", "2026-01-02", TRADE_DATE,
        )
        assert "FROZEN BENCHMARK" in prices
        assert "2026-01-09" in prices
        indicators = route_to_vendor(
            "get_indicators", "NVDA", "rsi", TRADE_DATE, 20,
        )
        assert "rsi values" in indicators
        from tradingagents.agents.tools import get_verified_market_snapshot

        snapshot = get_verified_market_snapshot.invoke({
            "symbol": "NVDA", "curr_date": TRADE_DATE,
            "trade_date": TRADE_DATE,
        })
        assert "Latest trading row used: 2026-01-09" in snapshot
        assert "rsi" in snapshot
        # A different ticker cannot silently reuse NVDA data.
        with pytest.raises(ValueError, match="only supports"):
            load_frozen_ohlcv("MSFT", TRADE_DATE)
        with pytest.raises(RuntimeError, match="blocks unsupported"):
            route_to_vendor("get_news", "NVDA", "2026-01-02", TRADE_DATE)


@pytest.mark.unit
def test_frozen_fixture_is_fail_closed(frozen_config, tmp_path):
    with run_config(frozen_config):
        with pytest.raises(Exception, match="does not cover"):
            route_to_vendor(
                "get_stock_data", "NVDA", "2020-01-01", TRADE_DATE,
            )
        file = tmp_path / "nvda-SYNTHETIC-TEST-ONLY.csv"
        file.write_text(file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="SHA-256"):
            load_frozen_ohlcv("NVDA", TRADE_DATE)


@pytest.mark.unit
def test_real_graph_with_scripted_llm_uses_frozen_market_only(
    frozen_config, monkeypatch, tmp_path,
):
    _disable_vendor_network(monkeypatch)
    model = ScriptedModel()
    monkeypatch.setattr(
        trading_graph, "create_tier_client",
        lambda config, tier, **kwargs: _Client(model),
    )
    graph = trading_graph.TradingAgentsGraph(
        selected_analysts=("market",), config=frozen_config,
    )
    report = collect_trajectory(
        graph, "NVDA", TRADE_DATE,
        case_id="SYNTHETIC_OFFLINE_E2E", attempt_id="01",
        output_dir=tmp_path / "traces",
    )
    assert report["status"] == "pass", report
    assert report["metrics"]["tool_calls"] == 3
    assert report["coverage"]["tool_call_linkage"]["matched"] == 3
    assert report["rules"]["model_tool_intent"]["status"] == "pass"
