"""历史决策结算：持仓观察期结束后，根据基准评估决策表现，
并将反思经验记录到 Memory Log。"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.market import get_closes

logger = logging.getLogger(__name__)


def resolve_benchmark(ticker: str, config: dict) -> str:
    """选择计算指定 ``ticker`` Alpha 时使用的基准。

    设置 ``config["benchmark_ticker"]`` 时优先使用它；否则按 ticker
    交易所后缀选择对应指数（如 ``.T`` 对应东京市场）。
    没有后缀的美股默认使用空后缀映射（通常为 SPY）。
    无法识别的后缀，包括 ``BRK.B`` 这类含点的美股代码，也使用该默认项。
    收益率均按百分比比较，各自在自己的币种中计算。"""

    explicit = config.get("benchmark_ticker")
    if explicit:
        # 基准与分析标的使用同一套别名映射；若别名无法解析，
        # 将找不到价格，导致相关决策一直待结算。
        return normalize_symbol(explicit)
    benchmark_map = config.get("benchmark_map", {})
    ticker_upper = normalize_symbol(ticker)
    for suffix, benchmark in benchmark_map.items():
        if suffix and ticker_upper.endswith(suffix.upper()):
            return benchmark
    return benchmark_map.get("", "SPY")


def _by_day(closes):
    """按交易日日期索引收盘价，剔除缺失值。

    Yahoo 日线通常标记为所在市场时区的零点（加密资产可能是 UTC，
    SPY 则为纽约时间），因此两个序列按日期而不是精确时刻对齐。
    收盘价为空或非正数都不能参与评分。"""
    closes = closes[closes > 0]
    if getattr(closes.index, "tz", None) is not None:
        closes = closes.tz_localize(None)
    return closes


def fetch_returns(
    ticker: str, trade_date: str, holding_days: int = 5,
    benchmark: str = "SPY",
) -> tuple[float | None, float | None, int | None, str | None]:
    """获取从 trade_date 之后开始、持有 holding_days 的原始收益与 Alpha。

    仓位在 ``trade_date`` 之后的第一个收盘价进入，
    因为该分析日可能包含当日收盘之后才发布的新闻（#1518）。
    ``benchmark`` 由调用者通过 ``resolve_benchmark`` 选择。
    返回 ``(raw_return, alpha_return, holding_days, resolution_date)``，
    其中 ``resolution_date`` 是最后一个参与计算的价格日期，
    即结果变得可知的时间（#1251）。
    若完整持仓窗口还没结束，则全部返回 None（#1169）。
    获取或读取价格失败时抛出异常，供上层报告。"""
    start = datetime.strptime(trade_date, "%Y-%m-%d")
    # holding_days 统计入场后的交易日数，因此请求行情时需估计对应自然日跨度
    # （每 5 个交易日约 7 个自然日），再额外增加一周以覆盖节假日。
    end = start + timedelta(days=round((holding_days + 1) * 7 / 5) + 7)
    end_str = end.strftime("%Y-%m-%d")

    # 取得分析实际使用的标的收盘价（如 XAUUSD → GC=F，#984），
    # 从分析日期之后第一个交易时段开始。
    stock = _by_day(get_closes(ticker, trade_date, end_str))
    stock = stock[stock.index > start]
    # 基准数据提前一周读取，确保入场日期之前或当天有可用收盘价。
    bench_start = (start - timedelta(days=7)).strftime("%Y-%m-%d")
    bench = _by_day(get_closes(benchmark, bench_start, end_str))

    # 必须等完整持仓观察期结束才能结算。
    # 到期前重新运行时仍保持待结算，下次继续尝试，
    # 不能提前以不完整收益率结算（#1169）。
    if len(stock) <= holding_days:
        return None, None, None, None
    entry, exit_ = stock.index[0], stock.index[holding_days]
    # 窗口由目标标的的交易日决定，基准使用各个窗口端点之前
    # 最近的收盘价，保证收益率对比的日期一致。
    # 即使标的与基准交易日历不同（加密资产周末交易、指数不交易），
    # 基准也必须有覆盖退出时点的价格，否则其收盘价可能仍会变化。
    if bench.empty or bench.index[0] > entry or bench.index[-1] < exit_:
        return None, None, None, None

    raw = float((stock.iloc[holding_days] - stock.iloc[0]) / stock.iloc[0])
    bench_entry, bench_exit = bench.asof(entry), bench.asof(exit_)
    bench_ret = float((bench_exit - bench_entry) / bench_entry)
    alpha = raw - bench_ret
    # 所有用于计算的收盘价在退出日期时都已经公开；
    # 该日期就是向历史运行注入经验时的时点过滤上限（#1251）。
    resolution_date = exit_.strftime("%Y-%m-%d")
    return raw, alpha, holding_days, resolution_date


@dataclass
class Settlement:
    """描述一次结算的结果：成功结算与未能结算的决策。"""

    settled: list[tuple[str, str]] = field(default_factory=list)        # (ticker, date)
    failed: list[tuple[str, str, str]] = field(default_factory=list)    # (ticker, date, reason)


def settle_pending(ticker: str, memory_log, reflector, config: dict, wait: bool = True) -> Settlement:
    """结算指定 ticker 中已经完成持仓观察期的待处理决策。

    同一日志同时仅允许一轮结算（``memory_log.settling()``），
    防止多个运行对同一决策重复请求反思而产生费用。
    每笔反思完成后立即写入，任务中断也能保留已完成部分。
    未到期记录跳过；行情获取失败或反思失败的记录保持待结算，
    并在结果中报告失败原因。``wait=False`` 不等待其他结算任务。"""
    result = Settlement()
    with memory_log.settling(wait=wait) as held:
        if not held:
            return result
        pending = [e for e in memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return result
        benchmark = resolve_benchmark(ticker, config)
        for entry in pending:
            try:
                raw, alpha, days, resolution_date = fetch_returns(
                    ticker, entry["date"], config.get("holding_period_days", 5), benchmark=benchmark,
                )
            except Exception as exc:
                logger.warning("Prices for settling %s on %s were unavailable: %s", ticker, entry["date"], exc)
                result.failed.append((ticker, entry["date"], f"prices unavailable: {exc}"))
                continue
            if raw is None:
                continue  # the holding window has not traded yet: try again next run
            try:
                reflection = reflector.reflect_on_final_decision(
                    final_decision=entry.get("decision", ""),
                    raw_return=raw,
                    alpha_return=alpha,
                    benchmark_name=benchmark,
                    holding_days=days,
                )
            except Exception as exc:
                # 反思需要调用 LLM 提供商；如果遇到短暂故障，
                # 保持记录待结算，让下次结算流程重试，而不是终止分析。
                logger.warning("Reflection failed for %s on %s: %s", ticker, entry["date"], exc)
                result.failed.append((ticker, entry["date"], f"reflection failed: {exc}"))
                continue
            if memory_log.update_with_outcome(ticker, entry["date"], raw, alpha, days, reflection,
                                              resolution_date=resolution_date):
                result.settled.append((ticker, entry["date"]))
    return result


def settle_all_pending(memory_log, reflector, config: dict, wait: bool = True) -> Settlement:
    """结算所有 ticker 中持仓观察期已经结束的待处理决策（#1445）。

    一次运行只会自动结算自己的 ticker；如果某个 ticker 停止被分析，
    其决策可能始终待结算，从而无法供后续运行复用对应经验。"""
    result = Settlement()
    for ticker in dict.fromkeys(e["ticker"] for e in memory_log.get_pending_entries()):
        done = settle_pending(ticker, memory_log, reflector, config, wait=wait)
        result.settled.extend(done.settled)
        result.failed.extend(done.failed)
    return result
