import logging
import os

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry

logger = logging.getLogger(__name__)

# 若最新 OHLCV 记录比请求日期早超过指定自然日数，则判定数据过期。
# 阈值应足以覆盖长假，又能发现 yfinance 偶尔返回的几年前旧数据（#1021）。
MAX_OHLCV_STALE_DAYS = 10

# 当日缓存尚未包含请求日期时可复用的 TTL（#1150）。
# 需要足够短，以便当日收盘数据公布后尽快刷新；
# 也不能短到节假日每次调用都触发下载。
OHLCV_CACHE_TTL_SECONDS = 900


def _ensure_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """将日期列统一命名为 ``Date``。

    部分 yfinance 版本的索引没有名称，reset_index() 后得到 index；
    日内数据也可能使用 Datetime。需要识别日期列并重命名，
    否则技术指标可能因为找不到 Date 而悄然缺失。"""
    if "Date" in data.columns:
        return data
    for candidate in ("index", "Datetime", "date"):
        if candidate in data.columns:
            return data.rename(columns={candidate: "Date"})
    return data


def _local_midnight(value) -> pd.Timestamp:
    """将单个时间戳转换为无时区、规范化到当地零点的日期；失败返回 NaT。"""
    if pd.isna(value):
        return pd.NaT
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)  # drop tz, keep the local wall-clock date
    return ts.normalize()


def _normalize_dates(dates) -> pd.Series:
    """将时间转换为无时区、当地零点的日期，确保带时区或日内时间戳
    能与无时区的 ``as_of_date`` 正确比较（#1201）。

    逐项转换是因为五年历史行情跨越夏令时变化，缓存 CSV 也可能保留
    不同 UTC 偏移。如果直接统一到 UTC，会把部分正时区市场
    移到上一日期；保留每个行情条目自己的本地日期可同时避免两类问题。"""
    return pd.to_datetime(pd.Series(dates).map(_local_midnight))


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """为 stockstats 规范化行情 DataFrame：解析日期，
    将价格转为数值，无效值置为 NaN。
    是否删除不完整行和填补缺口交给 ``_fill_price_gaps``，
    以便调用方先检查窗口内最新一条记录是否有效（#1201）。"""
    data = _ensure_date_column(data)
    data["Date"] = _normalize_dates(data["Date"])
    data = data.dropna(subset=["Date"]).copy()

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    return data


def _fill_price_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """删除没有收盘价的记录，对其他价格缺口前向或后向填充，
    使技术指标能够基于连续序列计算。"""
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    # 先 copy()，确保对筛选后数据的修改发生在副本上，而不是视图。
    data = data.dropna(subset=["Close"]).copy()
    data[price_cols] = data[price_cols].ffill().bfill()
    return data


def _coerce_ohlcv_dates(data: pd.DataFrame) -> pd.Series:
    """从 OHLCV 表中提取已解析的日期，无论 Date 位于列还是索引。"""
    if "Date" in data.columns:
        return pd.to_datetime(data["Date"], errors="coerce").dropna()
    # yfinance 常将日期保存在 DatetimeIndex 中，且索引可能没有名称。
    if isinstance(data.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(data.index, errors="coerce")).dropna()
    # 兜底：将索引转成列，再搜索任意类似日期的列。
    df = data.reset_index()
    for col in ("Date", "Datetime", "date", "index"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                return parsed
    return pd.Series(dtype="datetime64[ns]")


def _assert_ohlcv_not_stale(
    data: pd.DataFrame,
    as_of_date: str,
    symbol: str,
    canonical: str | None = None,
    *,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> None:
    """拒绝最新日期远早于 as_of_date 的过期 OHLCV。

    抛出带有具体过期信息的 NoMarketDataError，让路由器
    像其他“提供商无可用数据”情况一样尝试下一个数据源，
    最终返回明确的不可用标记。空表仍交给调用方原有逻辑处理；
    这里专门阻止看似有内容、实则是一年前价格的危险情况（#1021）。"""
    if data is None or data.empty:
        return
    requested = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(requested):
        return
    requested = requested.normalize()
    dates = _coerce_ohlcv_dates(data)
    if dates.empty:
        return
    latest = dates.max().normalize()
    stale_days = (requested - latest).days
    if stale_days > max_stale_days:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"latest row is {latest.date()}, {stale_days} days before the "
            f"requested {requested.date()} (stale) — refusing to use it",
        )


def _cache_is_fresh(data_file, as_of_dt, now) -> bool:
    """判断指定标的的缓存下载是否仍可用于当前请求。

    缓存保存的是写入当日下载的行情，因此一般只用于同一天。
    当日请求如果缓存超过 TTL 也要重新下载，因为 Yahoo 可能在交易中
    发布尚未定稿的日 K，且仅从 Close 字段无法判断它是否收盘（#1150）。"""
    written = pd.Timestamp.fromtimestamp(os.path.getmtime(data_file))
    if written.date() != now.date():
        return False
    return as_of_dt.date() < now.date() or (now - written).total_seconds() <= OHLCV_CACHE_TTL_SECONDS


def load_ohlcv(symbol: str, as_of_date: str, fill_gaps: bool = True) -> pd.DataFrame:
    """使用缓存获取 OHLCV，并过滤未来数据以避免回测前视偏差。

    下载截至今天的近五年数据，按标的缓存；后续请求优先复用缓存。
    读取时删除晚于 as_of_date 的记录，防止回测提前看到未来价格。
    ``fill_gaps`` 会用前值填补缺口，以维持指标计算所需的连续序列；
    设为 False 时返回供应商原始值，缺失记录仍保持为空。"""
    # 先将券商/外汇代码转换为 Yahoo 规则（如 XAUUSD+ → GC=F），
    # 再拒绝可能通过缓存文件名逃逸目录的值（如 ``../../tmp/x``）。
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)

    config = get_config()
    as_of_dt = pd.to_datetime(as_of_date).normalize()

    # 每个 ticker 一个缓存文件，保存最近五年至今的下载数据。
    now = pd.Timestamp.today()
    start_date = now - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    # yfinance 的 ``end`` 不包含右边界；若 as_of_date 是今天，
    # 需请求到明天才能得到今天的数据（#986）。
    # 之后仍会用 as_of_date 截止日期过滤未来数据。
    end_str = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-YFin-data.csv",
    )

    # 前一次请求失败（未知标的、暂时限流）可能留下空缓存。
    # 空文件或没有数据列时应视为缓存未命中并重新抓取，
    # 不能一直复用受污染的缓存。
    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if (
            not cached.empty
            and "Close" in cached.columns
            and _cache_is_fresh(data_file, as_of_dt, now)
        ):
            data = cached

    if data is None:
        # yf.download 会把包括限流在内的错误捕获并返回空表；
        # Ticker.history 则会抛出限流异常，便于进行重试。
        downloaded = yf_retry(lambda: yf.Ticker(canonical).history(
            start=start_str,
            end=end_str,
            auto_adjust=True,
            actions=False,
        ))
        if downloaded is None:
            raise_for_empty(symbol, canonical, "price rows")
        downloaded = _ensure_date_column(downloaded.reset_index())
        # 只缓存真实有效数据，不能持久化空表。
        if downloaded.empty or "Close" not in downloaded.columns:
            raise_for_empty(symbol, canonical, "price rows")
        replace_file(data_file, lambda temp: downloaded.to_csv(temp, index=False, encoding="utf-8"))
        data = downloaded

    data = _clean_dataframe(data)

    # 按 as_of_date 截取数据，防止回测前视偏差。
    data = data[data["Date"] <= as_of_dt]

    # 最新 K 线缺少 Close 表示该交易时段尚未定稿，
    # 不意味着标的完全无数据。下方 _fill_price_gaps 会删除此类记录，
    # 包括序列中间的缺收盘价行；只有整个范围都没有收盘价才算无数据
    # （#1201、#1289）。
    if not data.empty and pd.isna(data["Close"].iloc[-1]):
        settled = data["Close"].notna().to_numpy().nonzero()[0]
        if settled.size == 0:
            raise NoMarketDataError(
                symbol, canonical, "no bar in range has a closing price"
            )
        logger.warning(
            "%s: %d trailing bar(s) through %s have no closing price; using %s "
            "as the latest close.", canonical, len(data) - settled[-1] - 1,
            data["Date"].iloc[-1].date(), data["Date"].iloc[settled[-1]].date(),
        )

    # 技术指标需要连续价格序列，因此计算指标时可使用前值填补缺口。
    # 但如果调用方需要报告真实报价，则必须提供原始数据，
    # 不能把上一交易日价格当作本交易日价格。
    data = _fill_price_gaps(data) if fill_gaps else data.dropna(subset=["Close"]).copy()

    # 拒绝最新记录远早于 as_of_date 的过期行情，
    # 避免把一年前价格作为当前技术指标输入（#1021）。
    _assert_ohlcv_not_stale(data, as_of_date, symbol, canonical)

    return data


