"""Offline checks for default Chinese news routing and point-in-time filtering."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors import cn_news


@pytest.fixture
def mock_cn_feeds(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_info_global_cls=lambda: pd.DataFrame([
            {
                "标题": "阿里巴巴宣布最新业务进展",
                "内容": "阿里巴巴集团公布了业务消息。",
                "发布日期": "2026-11-04",
                "发布时间": "11:30:00",
            },
            {
                "标题": "阿里巴巴未来消息",
                "内容": "future",
                "发布日期": "2026-11-05",
                "发布时间": "10:00:00",
            },
        ]),
        stock_info_global_em=lambda: pd.DataFrame([
            {"标题": "央行市场消息", "内容": "宏观动态",
             "发布时间": "2026-11-04 09:10:00", "链接": "https://example.com/1"}
        ]),
        stock_info_global_sina=lambda: pd.DataFrame(),
    ))


@pytest.mark.unit
def test_baba_domestic_news_filters_future_items(mock_cn_feeds):
    out = cn_news.get_news("BABA", "2026-11-03", "2026-11-04")
    assert "阿里巴巴宣布最新业务进展" in out
    assert "2026-11-05" not in out
    assert "Source: 财联社" in out
    assert "央行市场消息" not in out


@pytest.mark.unit
def test_hk_alibaba_alias_is_supported(mock_cn_feeds):
    out = cn_news.get_news("9988.HK", "2026-11-03", "2026-11-04")
    assert "阿里巴巴宣布最新业务进展" in out


@pytest.mark.unit
def test_global_news_includes_only_dated_window(mock_cn_feeds):
    out = cn_news.get_global_news("2026-11-04", look_back_days=1, limit=3)
    assert "阿里巴巴宣布最新业务进展" in out
    assert "央行市场消息" in out
    assert "阿里巴巴未来消息" not in out


@pytest.mark.unit
def test_news_has_no_historical_coverage_falls_back(mock_cn_feeds):
    with pytest.raises(VendorUnavailableError, match="no verified"):
        cn_news.get_news("BABA", "2026-10-27", "2026-10-28")


@pytest.mark.unit
def test_unavailable_optional_akshare_is_a_vendor_failure(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "akshare", None)
    with pytest.raises(VendorUnavailableError, match="requires AKShare"):
        cn_news.get_news("BABA", "2026-11-03", "2026-11-04")


@pytest.mark.unit
def test_router_vendor_order_prioritises_cn_news():
    from tradingagents.dataflows.router import VENDOR_METHODS
    from tradingagents.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["data_vendors"]["news_data"] == "cn_news,yfinance"
    assert "cn_news" in VENDOR_METHODS["get_news"]
    assert "cn_news" in VENDOR_METHODS["get_global_news"]
