"""最近交易日的 K 线不能悄然消失（#1201）。

yfinance 可能返回 Close 为 NaN 的最新 K 线，表示交易尚未结算或数据异常。
旧路径先丢弃缺收盘价行，导致前一交易日被误认为最新。
现在先规范化日期再过滤，保留最近已结算 K 线，不能虚构收盘价。
但也不应直接拒绝整组行情，否则会把有效标的误报无效（#1289）；
只有整个窗口都缺收盘价时才视为无数据，剩余数据仍需检查过期。"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.yahoo import ohlcv


def _stamp(path, ts):
    """缓存修改时间使用本地时间 ``ts``，读取时也按本地时间解释；
    无时区 pd.Timestamp.timestamp() 可能被错误视为 UTC。"""
    t = ts.to_pydatetime().timestamp()
    os.utime(path, (t, t))

# --- date normalization -----------------------------------------------------


@pytest.mark.unit
def test_normalize_dates_strips_tz_and_normalizes_to_midnight():
    aware = pd.Series(pd.to_datetime(
        ["2026-05-08 09:30:00-04:00", "2026-05-09 16:00:00-04:00"]
    ))
    out = ohlcv._normalize_dates(aware)
    assert out.dt.tz is None
    assert list(out) == [pd.Timestamp("2026-05-08"), pd.Timestamp("2026-05-09")]


@pytest.mark.unit
def test_normalize_dates_leaves_naive_dates_at_midnight():
    naive = pd.Series(pd.to_datetime(["2026-05-08 14:30:00", "2026-05-09 00:00:00"]))
    out = ohlcv._normalize_dates(naive)
    assert out.dt.tz is None
    assert list(out) == [pd.Timestamp("2026-05-08"), pd.Timestamp("2026-05-09")]


@pytest.mark.unit
def test_normalize_dates_handles_mixed_dst_offsets():
    # 五年美股 K 线跨越夏令时变化，缓存 CSV 可能含不同 UTC 偏移；
    # 每条记录需保留自身市场本地日期。
    mixed = pd.Series([
        "2026-01-08 00:00:00-05:00",  # EST
        "2026-06-08 00:00:00-04:00",  # EDT
        "not-a-date",                 # -> NaT
    ])
    out = ohlcv._normalize_dates(mixed)
    assert out.iloc[0] == pd.Timestamp("2026-01-08")
    assert out.iloc[1] == pd.Timestamp("2026-06-08")
    assert pd.isna(out.iloc[2])


@pytest.mark.unit
def test_normalize_dates_keeps_positive_offset_local_date():
    # 东京市场 +09:00 的午夜 K 线必须保留当地日期；
    # 直接转换 UTC 会错误地归入前一天。
    jst = pd.Series(["2026-05-08 00:00:00+09:00"])
    assert ohlcv._normalize_dates(jst).iloc[0] == pd.Timestamp("2026-05-08")


# --- fill vs guard responsibilities ----------------------------------------

@pytest.mark.unit
def test_clean_dataframe_keeps_nan_close_for_the_caller_to_inspect():
    # _clean_dataframe 只规范化数据，不直接删除 NaN 收盘价。
    df = pd.DataFrame({"Date": ["2026-05-08", "2026-05-09"], "Close": [100.0, float("nan")]})
    cleaned = ohlcv._clean_dataframe(df)
    assert len(cleaned) == 2
    assert pd.isna(cleaned["Close"].iloc[-1])


@pytest.mark.unit
def test_fill_price_gaps_drops_nan_close_rows():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-05-07", "2026-05-08"]),
                       "Close": [float("nan"), 100.0]})
    filled = ohlcv._fill_price_gaps(df)
    assert len(filled) == 1
    assert filled["Close"].iloc[0] == 100.0


# --- load_ohlcv end-to-end (with a mocked cache read) -----------------------

def _run_load(monkeypatch, tmp_path, frame, curr_date):
    """预先写入缓存 DataFrame 后运行 load_ohlcv，不使用网络。"""
    monkeypatch.setattr(ohlcv, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    today = pd.Timestamp(curr_date)
    monkeypatch.setattr(ohlcv.pd.Timestamp, "today", staticmethod(lambda: today))
    cache_file = tmp_path / "AAPL-YFin-data.csv"
    cache_file.write_text(frame.to_csv(index=False))
    _stamp(cache_file, today)

    def _fail_download(*a, **k):
        raise AssertionError("should use the seeded cache, not download")
    monkeypatch.setattr(ohlcv.yf, "Ticker", lambda symbol: SimpleNamespace(history=_fail_download))
    return ohlcv.load_ohlcv("AAPL", curr_date)


@pytest.mark.unit
def test_unsettled_latest_bar_is_served_as_the_last_settled_bar(monkeypatch, tmp_path):
    # curr_date 当天无收盘价时，返回最近已结算 K 线，
    # 不能把整个标的视为不可用（#1289）。
    frame = pd.DataFrame({
        "Date": ["2026-05-07", "2026-05-08"],
        "Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0],
        "Close": [100.5, float("nan")], "Volume": [1_000_000, 1_000_000],
    })
    out = _run_load(monkeypatch, tmp_path, frame, "2026-05-08")
    assert out["Date"].iloc[-1] == pd.Timestamp("2026-05-07")
    assert out["Close"].iloc[-1] == 100.5


@pytest.mark.unit
def test_no_settled_bar_at_all_is_still_no_data(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "Date": ["2026-05-07", "2026-05-08"],
        "Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0],
        "Close": [float("nan"), float("nan")], "Volume": [1_000_000, 1_000_000],
    })
    with pytest.raises(NoMarketDataError, match="no bar in range has a closing price"):
        _run_load(monkeypatch, tmp_path, frame, "2026-05-08")


@pytest.mark.unit
def test_serving_the_last_settled_bar_does_not_bypass_the_staleness_check(
    monkeypatch, tmp_path
):
    # 删除缺收盘价的末尾记录后，仍要检查剩余数据是否过期，
    # 不能使多年未更新的行情重新变成“有效”。
    frame = pd.DataFrame({
        "Date": ["2026-01-05", "2026-05-08"],
        "Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0],
        "Close": [100.5, float("nan")], "Volume": [1_000_000, 1_000_000],
    })
    with pytest.raises(NoMarketDataError, match="stale"):
        _run_load(monkeypatch, tmp_path, frame, "2026-05-08")


@pytest.mark.unit
def test_older_nan_close_row_is_still_dropped(monkeypatch, tmp_path):
    # 删除序列中间的过期缺口，保留有效的最新 K 线。
    frame = pd.DataFrame({
        "Date": ["2026-05-06", "2026-05-07", "2026-05-08"],
        "Open": [100.0, 101.0, 102.0], "High": [101.0, 102.0, 103.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [100.5, float("nan"), 102.5], "Volume": [1_000_000, 1_000_000, 1_000_000],
    })
    out = _run_load(monkeypatch, tmp_path, frame, "2026-05-08")
    assert out["Close"].iloc[-1] == 102.5
    assert (out["Date"] == pd.Timestamp("2026-05-07")).sum() == 0  # the NaN row is gone


@pytest.mark.unit
def test_tz_aware_latest_bar_is_kept_at_the_cutoff(monkeypatch, tmp_path):
    # 截止日内带时区或日内时间戳的最新 K 线，
    # 不能因日期比较方式不一致而被错误排除。
    frame = pd.DataFrame({
        "Date": ["2026-05-07 09:30:00-04:00", "2026-05-08 09:30:00-04:00"],
        "Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0],
        "Close": [100.5, 101.5], "Volume": [1_000_000, 1_000_000],
    })
    out = _run_load(monkeypatch, tmp_path, frame, "2026-05-08")
    assert out["Close"].iloc[-1] == 101.5
    assert out["Date"].iloc[-1] == pd.Timestamp("2026-05-08")


@pytest.mark.unit
def test_the_snapshot_does_not_present_a_filled_price_as_reported(monkeypatch, tmp_path):
    """缺口填补是为了让指标在连续价格序列上计算；
    核验快照必须显示数据源真实报价，不能由防止虚构价格的模块反过来制造价格。"""
    from tradingagents.dataflows.vendors.yahoo import ohlcv, snapshot

    frame = pd.DataFrame({
        "Date": ["2026-05-06", "2026-05-07", "2026-05-08"],
        "Open": [100.0, 104.5, ""],     # the latest bar has not settled
        "High": [101.0, 105.5, ""],
        "Low": [99.0, 103.5, ""],
        "Close": [100.5, 105.0, 106.0],
        "Volume": [1000000, 1000000, ""],
    })
    today = pd.Timestamp("2026-05-08 12:00")
    monkeypatch.setattr(ohlcv, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(ohlcv.pd.Timestamp, "today", staticmethod(lambda: today))
    cache = tmp_path / "AAPL-YFin-data.csv"
    cache.write_text(frame.to_csv(index=False))
    _stamp(cache, today)
    monkeypatch.setattr(ohlcv.yf, "Ticker", lambda symbol: SimpleNamespace(
        history=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should read the seeded cache"))))

    out = snapshot.build_verified_market_snapshot("AAPL", "2026-05-08", 3)

    row = out.split("Latest verified OHLCV row")[1].split("###")[0]
    assert "104.50" not in row and "105.50" not in row  # the previous session's numbers
    assert "106.00" in row  # the close the vendor did report
