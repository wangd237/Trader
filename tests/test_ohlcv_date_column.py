"""测试 Yahoo OHLCV 加载器兼容非 `Date` 命名的日期列（#890）。

行情的日期列有时叫 `index` 或 `Datetime`，
若未规范化，技术指标可能静默失效。"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.yahoo import ohlcv


def _ohlcv(date_col: str) -> pd.DataFrame:
    """构造日期列名为 `date_col` 的 OHLCV 数据框。"""
    dates = pd.bdate_range("2026-04-01", periods=10)
    return pd.DataFrame({
        date_col: dates,
        "Open": [100.0 + i for i in range(10)],
        "High": [101.0 + i for i in range(10)],
        "Low": [99.0 + i for i in range(10)],
        "Close": [100.5 + i for i in range(10)],
        "Volume": [1_000_000 + i for i in range(10)],
    })


@pytest.mark.unit
class TestEnsureDateColumn:
    def test_renames_index_column(self):
        out = ohlcv._ensure_date_column(_ohlcv("index"))
        assert "Date" in out.columns and "index" not in out.columns

    def test_renames_datetime_and_date_variants(self):
        assert "Date" in ohlcv._ensure_date_column(_ohlcv("Datetime")).columns
        assert "Date" in ohlcv._ensure_date_column(_ohlcv("date")).columns

    def test_leaves_existing_date_untouched(self):
        df = _ohlcv("Date")
        assert ohlcv._ensure_date_column(df) is df  # no-op short-circuit

    def test_no_datelike_column_is_left_alone(self):
        df = pd.DataFrame({"Close": [1, 2, 3]})
        out = ohlcv._ensure_date_column(df)
        assert "Date" not in out.columns  # nothing to rename; caller handles


@pytest.mark.unit
class TestCleanDataframeAcrossVersions:
    def test_clean_handles_index_column(self):
        """日期列即使叫 `index` 也应规范化为可用日期，
        不能抛出 KeyError: 'Date'。"""
        cleaned = ohlcv._clean_dataframe(_ohlcv("index"))
        assert "Date" in cleaned.columns
        assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
        assert len(cleaned) == 10

    def test_clean_handles_legacy_date_column(self):
        cleaned = ohlcv._clean_dataframe(_ohlcv("Date"))
        assert len(cleaned) == 10

    def test_indicators_compute_after_index_rename(self):
        """stockstats 应能处理初始日期列为 `index` 的行情，
        不能在计算指标时不断报错。"""
        from stockstats import wrap
        cleaned = ohlcv._clean_dataframe(_ohlcv("index"))
        df = wrap(cleaned)
        df["close_5_sma"]  # triggers calculation
        assert "close_5_sma" in df.columns
        assert df["close_5_sma"].notna().any()


@pytest.mark.unit
def test_cleaning_a_frame_with_undated_rows_writes_to_its_own_copy():
    raw = pd.DataFrame({"Date": ["2026-01-08", None, "2026-01-09"], "Close": ["1", "2", "x"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cleaned = ohlcv._clean_dataframe(raw)
    assert cleaned["Close"].tolist()[0] == 1.0
