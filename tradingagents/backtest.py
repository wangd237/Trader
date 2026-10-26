"""在多个 ticker 与日期组成的网格上执行 Graph，并评价运行结果。

单次运行只生成一个决策，无法判断系统整体决策质量。
因此对多个 (ticker, date) 组合运行同一条分析流程并聚合结果。
Memory Log 可作为结果表：每次运行已经保存评级，随后结算实际收益率
以及相对于地区基准指数的 Alpha，因此不必额外存一份结果。

本模块只评价决策质量，不是持仓组合模拟器，也不应该扩大为模拟器。
要将评级转换成真实成交订单，还需要数量、成交价格和资金账户等信息。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import TradingMemoryLog

logger = logging.getLogger(__name__)


def iter_grid(start_date: str, end_date: str, every_n_days: int = 1) -> list[str]:
    """从 ``start_date`` 开始生成分析日期，但绝不超过今天。
    
    未来日期没有可用于结算的结果，Graph 也拒绝执行，
    因此网格在今天停止，避免生成无法评分的任务。"""
    start, end = _canonical(start_date), _canonical(end_date)
    if every_n_days < 1:
        raise ValueError("every_n_days must be at least 1")
    if end < start:
        raise ValueError(f"the grid ends before it starts: {end_date} is before {start_date}")

    last = min(end, datetime.strptime(get_current_date(), "%Y-%m-%d"))
    dates, cursor = [], start
    while cursor <= last:
        dates.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=every_n_days)
    return dates


def _canonical(date: str) -> datetime:
    """解析网格日期边界；凡是单次运行日期校验会拒绝的输入，这里也应拒绝。"""
    try:
        parsed = datetime.strptime(str(date), "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid dates must be in YYYY-MM-DD format, got {date!r}") from exc
    if parsed.strftime("%Y-%m-%d") != str(date):
        raise ValueError(f"grid dates must be in YYYY-MM-DD format, got {date!r}")
    return parsed


def _alpha(entry: dict) -> float | None:
    """返回已结算记录的 Alpha 收益率；未结算时返回 None。
    
    日志中的百分比只保留一位小数，因此这里的聚合精度为
    0.1 个百分点，而非原始行情精度。"""
    text = (entry.get("alpha") or "").strip().rstrip("%")
    try:
        return float(text) / 100
    except ValueError:
        return None


@dataclass
class BacktestResult:
    run_id: str
    log_path: Path
    cells_run: int = 0
    skipped: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    settlement_failures: list[tuple[str, str]] = field(default_factory=list)


# 各评级所表达的涨跌方向预期，用于和真实结果比较。
# Hold 不预测方向，因此不能单凭 Alpha 判断它正确或错误。
_DIRECTION = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}


@dataclass
class RatingScore:
    count: int
    hit_rate: float | None
    mean_alpha: float


@dataclass
class BacktestSummary:
    resolved: int
    pending: int
    by_rating: dict[str, RatingScore]
    unscored: int = 0
    holding: str = ""

    def render(self) -> str:
        lines = [f"Resolved cells: {self.resolved} · pending: {self.pending}"
                 + (f" · unscored: {self.unscored}" if self.unscored else "")]
        for rating, score in self.by_rating.items():
            called = (f"called the direction {score.hit_rate:.0%}"
                      if score.hit_rate is not None else "no direction claimed")
            lines.append(
                f"- {rating}: n={score.count}, {called}, "
                f"mean alpha {score.mean_alpha:+.2%} vs the benchmark"
            )
        lines.append("")
        if self.pending:
            lines.append("Pending cells are not scored above; re-run to settle them.")
        lines.append(
            f"Alpha is measured over {self.holding} from the first close after each analysis date. "
            "One model sampling per cell, and text feeds are not archived, so "
            "these figures are indicative rather than repeatable."
        )
        return "\n".join(lines)


def run_backtest(
    tickers: list[str],
    dates: list[str],
    config: dict,
    asset_type: str = "stock",
    portfolio=None,
    selected_analysts=("market", "social", "news", "fundamentals"),
    run_id: str | None = None,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> BacktestResult:
    """对网格中的每个 ticker 和日期执行分析，写入本次独立的 Memory Log。
    
    不能污染实时运行使用的日志，否则批量回测会向未来真实运行注入
    大量历史上下文。已存在于本次日志的网格单元会跳过，
    因此中断后重新执行即可继续。在每个实际执行的单元开始前调用
    ``progress(done, total, ticker, date)``。"""
    # run_id 会成为目录组件，因此必须像 ticker 一样校验；
    # 绝对路径或只含点的值可能使输出逃逸到 results_dir 之外。
    run_id = safe_ticker_component(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(config["results_dir"]) / "backtest" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**config, "results_dir": str(run_dir),
                  "memory_log_path": str(run_dir / "trading_memory.md")}

    graph = TradingAgentsGraph(selected_analysts, config=run_config)
    result = BacktestResult(run_id=run_id, log_path=Path(run_config["memory_log_path"]))
    done = {(e["ticker"], e["date"]) for e in graph.memory_log.load_entries()}

    # 重复出现的 ticker 或日期只算一个网格单元，仅执行和结算一次。
    tickers, dates = list(dict.fromkeys(tickers)), list(dict.fromkeys(dates))
    cells = [(ticker, date) for ticker in tickers for date in dates]
    todo = [cell for cell in cells if cell not in done]
    result.skipped = len(cells) - len(todo)
    for index, (ticker, date) in enumerate(todo, 1):
        if progress:
            progress(index, len(todo), ticker, date)
        try:
            graph.propagate(ticker, date, asset_type, portfolio=portfolio)
            result.cells_run += 1
        except Exception as exc:  # one unreachable vendor must not end the sweep
            logger.warning("Backtest cell %s %s failed: %s", ticker, date, exc)
            result.failures.append((ticker, date, str(exc)))

    # 结算发生在同一 ticker 的下一次运行开始时，
    # 因此还需显式结算每个 ticker 的最后一个网格单元。
    for ticker in tickers:
        try:
            graph.settle_pending(ticker)
        except Exception as exc:  # reflection calls an LLM; one failure is not the sweep's
            logger.warning("Settling %s failed: %s", ticker, exc)
            result.settlement_failures.append((ticker, str(exc)))
    return result


def summarize(source: BacktestResult | str | Path) -> BacktestSummary:
    """按评级统计和评估回测（或指定 Memory Log）中已经结算的决策。"""
    if isinstance(source, BacktestResult):
        path = source.log_path      # a run whose cells all failed wrote no log: nothing to score
    elif Path(source).is_file():
        path = Path(source)
    else:
        raise FileNotFoundError(f"no memory log at {source}")
    entries = TradingMemoryLog({"memory_log_path": str(path)}).load_entries()
    # 无法解析评级的决策没有明确方向，既不能证明系统正确，
    # 也不能证明错误，应单独报告为“未评分”。
    resolved = [(e, _alpha(e)) for e in entries
                if not e["pending"] and e["rating"] != RATING_REVIEW]
    resolved = [(e, a) for e, a in resolved if a is not None]
    by_rating: dict[str, RatingScore] = {}
    for rating in dict.fromkeys(e["rating"] for e, _ in resolved):
        alphas = [a for e, a in resolved if e["rating"] == rating]
        direction = _DIRECTION.get(rating, 0)
        by_rating[rating] = RatingScore(
            count=len(alphas),
            hit_rate=(sum(a * direction > 0 for a in alphas) / len(alphas)) if direction else None,
            mean_alpha=sum(alphas) / len(alphas),
        )
    unscored = sum(1 for e in entries if e["rating"] == RATING_REVIEW)
    # 根据日志报告真实结果实际使用的观察期。
    windows = {f"{e['holding'][:-1]} trading days" for e, _ in resolved
               if (e.get("holding") or "").endswith("d")}
    return BacktestSummary(resolved=len(resolved),
                           pending=len(entries) - len(resolved) - unscored,
                           by_rating=by_rating, unscored=unscored,
                           holding=", ".join(sorted(windows)) or "the configured window")
