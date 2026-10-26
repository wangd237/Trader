import hashlib
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import tradingagents
from tradingagents.agents.context import build_instrument_context, resolve_instrument_identity
from tradingagents.agents.rating import run_rating
from tradingagents.dataflows.config import run_config, run_config_context, set_config
from tradingagents.dataflows.date_window import get_current_date, is_historical
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_tier_client, tier_provider
from tradingagents.memory import TradingMemoryLog, settlement
from tradingagents.memory.reflection import Reflector
from tradingagents.reporting import write_report_tree

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .setup import GraphSetup

logger = logging.getLogger(__name__)


def _validate_trade_date(trade_date) -> str:
    """将运行日期规范化为不晚于今天的 ``YYYY-MM-DD`` 字符串。"""
    value = str(trade_date)
    try:
        canonical = datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        canonical = False
    if not canonical:
        raise ValueError(f"trade_date must be a date in YYYY-MM-DD format, got {trade_date!r}")
    if value > get_current_date():
        raise ValueError(f"trade_date cannot be in the future: {value}")
    return value


# 不会改变分析结果的配置项：文件保存位置、是否启用检查点、
# 以及提供商失败后的重试次数。
_NOT_IN_SIGNATURE = frozenset({
    "results_dir", "data_cache_dir", "memory_log_path", "checkpoint_enabled", "llm_max_retries",
})


class TradingAgentsGraph:
    """交易 Agent 框架的核心编排类。"""

    def __init__(
        self,
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=False,
        config: dict[str, Any] = None,
        callbacks: list | None = None,
    ):
        """初始化交易 Agent Graph 及其组件。

        参数：
            selected_analysts：需要启用的分析师类型列表。
            debug：是否启用调试模式。
            config：配置字典，传入 None 时使用默认配置。
            callbacks：可选的回调处理器列表（例如跟踪 LLM 或工具调用统计）。"""
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        set_config(self.config)

        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        extra = {"callbacks": self.callbacks} if self.callbacks else {}
        self.deep_thinking_llm = create_tier_client(self.config, "deep", **extra).get_llm()
        self.quick_thinking_llm = create_tier_client(self.config, "quick", **extra).get_llm()

        self.memory_log = TradingMemoryLog(self.config)

        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        # 分析师每轮工具调用占用两个图步骤，另外还包括首次响应与报告收尾；
        # 若实际所需步骤数超过递归上限，运行会在此被终止。
        max_tool_rounds, max_recur_limit = self.config["max_tool_rounds"], self.config["max_recur_limit"]
        if 2 * max_tool_rounds + 2 >= max_recur_limit:
            raise ValueError(
                f"max_tool_rounds={max_tool_rounds} needs max_recur_limit above {2 * max_tool_rounds + 2}"
            )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.conditional_logic,
            max_tool_rounds,
        )

        self.propagator = Propagator(
            max_recur_limit=max_recur_limit,
        )
        self.reflector = Reflector(self.quick_thinking_llm)

        # 保存会影响图结构的运行选项，用于生成检查点签名。
        self.selected_analysts = tuple(selected_analysts)

        # 构建 Graph，同时保留 workflow，便于接入 checkpointer 后重新编译。
        self.workflow = self.graph_setup.setup_graph(selected_analysts, memory_node=self._memory_step)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None
        self._resuming = False

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock",
                                   trade_date: str | None = None) -> str:
        """一次性解析 ticker 的真实标的身份，并返回完整标的上下文。

        通过确定性的 yfinance 查询（带缓存、失败时不中断）构造上下文，
        让各 Agent 依据真实公司身份分析，避免仅从价格图中臆测公司（#814）。
        ``propagate()`` 和 CLI 都调用这里，保证不同入口都能向整个图传递标的身份。"""
        identity = resolve_instrument_identity(ticker)
        return build_instrument_context(ticker, asset_type, identity, trade_date)

    def _memory_as_of(self, trade_date) -> str | None:
        """确定历史经验可用性的时间截止点（#1251）。

        历史分析或回测（交易日早于今天）只使用该交易日之前已结算的经验。
        当日运行返回 None，即关闭过滤，使实时运行和迁移前没有结算日期的
        旧记录不受影响。"""
        return str(trade_date) if is_historical(trade_date) else None

    def _run_signature(self, asset_type: str, portfolio=None) -> str:
        """返回改变后必须使现有检查点失效的运行参数。

        这些参数参与生成检查点 thread ID，因此分析师组合、辩论和风险深度、
        资产模式变化后会从头运行，避免悄然沿用旧图（#1089）。
        其余配置（提供商、模型、端点、语言、数据源、上限）也必须参与，
        防止恢复到其他配置生成的报告；仅排除文件保存路径和重试行为。"""
        settings = {k: v for k, v in self.config.items() if k not in _NOT_IN_SIGNATURE}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return "|".join([
            "analysts=" + ",".join(self.selected_analysts),
            f"debate={self.config['max_debate_rounds']}",
            f"risk={self.config['max_risk_discuss_rounds']}",
            f"asset={asset_type}",
            # None、空仓位记录以及发生变化的仓位记录应视为三种不同运行。
            f"portfolio={portfolio.fingerprint() if portfolio is not None else 'none'}",
            # 图拓扑本身也会变化：如果旧检查点来自分析师串行执行的图，
            # 其中可能存在当前并行图不再拥有的待执行节点；
            # 如果检查点建立于 Memory Log 节点引入之前，恢复后将缺少历史经验。
            "analysts=parallel",
            "memory=parallel",
            f"settings={digest}",
        ])

    def propagate(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """针对指定公司和日期执行交易 Agent Graph。

        ``asset_type`` 用来切换默认股票流程和 #567 新增的加密资产流程
        （``"crypto"``）。CLI 根据 ticker 自动识别，程序调用者则需显式传入。
        配置启用 ``checkpoint_enabled`` 时，图会使用每个 ticker 独立的
        SqliteSaver 重新编译；相同 ticker 和日期的后续运行可从上次成功节点恢复。

        返回 ``(final_state, signal)``，其中 ``signal`` 为五级评级
        （Buy / Overweight / Hold / Underweight / Sell），或在无法解析评级时
        返回 ``"REVIEW"``（#1170）。映射到 PortfolioRating 枚举前应先调用
        ``tradingagents.agents.rating.is_review`` 进行判断。"""
        trade_date = _validate_trade_date(trade_date)

        with run_config(self.config), \
                self.checkpoint_scope(company_name, trade_date, asset_type, portfolio) as thread_id_value:
            return self._run_graph(
                company_name, trade_date, asset_type=asset_type,
                checkpoint_thread_id=thread_id_value, portfolio=portfolio,
            )

    def begin_checkpoint(self, company_name, trade_date, asset_type: str = "stock", portfolio=None) -> str | None:
        """使用每个 ticker 专属的 checkpointer 重新编译图，并返回要注入
        stream/invoke ``config`` 的 ``thread_id``；禁用检查点时返回 None。

        每次调用后都必须在 ``finally`` 中执行 :meth:`end_checkpoint`。
        ``propagate``（通过 :meth:`checkpoint_scope`）和 CLI 流式入口均使用
        这套机制，确保 ``--checkpoint`` 真正支持恢复（#1249）。
        以前只有 ``propagate`` 配置了检查点，CLI 使用普通图，导致标志无效。"""
        self._resuming = False
        if not self.config.get("checkpoint_enabled"):
            return None
        signature = self._run_signature(asset_type, portfolio)
        self._checkpointer_ctx = get_checkpointer(self.config["data_cache_dir"], company_name)
        saver = self._checkpointer_ctx.__enter__()
        self.graph = self.workflow.compile(checkpointer=saver)

        step = checkpoint_step(
            self.config["data_cache_dir"], company_name, str(trade_date), signature
        )
        self._resuming = step is not None
        if step is not None:
            logger.info("Resuming from step %d for %s on %s", step, company_name, trade_date)
        else:
            logger.info("Starting fresh for %s on %s", company_name, trade_date)
        return thread_id(company_name, str(trade_date), signature)

    def checkpoint_input(self, init_state):
        """返回流式执行或调用图时需要传入的值：从检查点恢复时为 None，
        全新运行时为初始状态。

        LangGraph 在传入 None 时恢复中断的线程；如果重新传入初始状态，
        消息 reducer 会重复追加这些消息，造成恢复后的状态包含重复消息（#1249）。"""
        return None if self._resuming else init_state

    def end_checkpoint(self):
        """检查点运行结束后，恢复不带 checkpointer 的普通 Graph。"""
        if self._checkpointer_ctx is not None:
            self._checkpointer_ctx.__exit__(None, None, None)
            self._checkpointer_ctx = None
            self.graph = self.workflow.compile()
        self._resuming = False

    @contextmanager
    def checkpoint_scope(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """供 propagate 路径使用的 begin/end_checkpoint 上下文管理器。"""
        try:
            yield self.begin_checkpoint(company_name, trade_date, asset_type, portfolio)
        finally:
            self.end_checkpoint()

    def clear_checkpoint_on_success(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """删除已完成运行的检查点，保证后续运行重新开始（#1249）。"""
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date),
                self._run_signature(asset_type, portfolio),
            )

    def run_settings(self) -> dict:
        """生成保存到报告和状态日志中的运行来源信息。

        只记录白名单字段；端点（backend_url 可能包含凭据）、密钥和本地路径
        一律不写入。"""
        cfg = self.config
        return {
            "version": tradingagents.__version__,
            "llm_provider": cfg.get("llm_provider"),
            "deep_think_provider": tier_provider(cfg, "deep") if cfg.get("llm_provider") else None,
            "deep_think_llm": cfg.get("deep_think_llm"),
            "quick_think_provider": tier_provider(cfg, "quick") if cfg.get("llm_provider") else None,
            "quick_think_llm": cfg.get("quick_think_llm"),
            "analysts": list(self.selected_analysts),
            "max_debate_rounds": cfg.get("max_debate_rounds"),
            "max_risk_discuss_rounds": cfg.get("max_risk_discuss_rounds"),
            "output_language": cfg.get("output_language"),
            "data_vendors": dict(cfg.get("data_vendors") or {}),
            "tool_vendors": dict(cfg.get("tool_vendors") or {}),
        }

    def save_reports(self, final_state, ticker, save_path=None, html=True) -> Path:
        """与 CLI 一样，为已完成的运行保存整套报告文件。

        程序调用者也会得到与 CLI 相同的磁盘报告；可以显式指定
        ``save_path``，否则默认写入 ``results_dir``。
        除非 ``html`` 为 False，否则还会生成一个 HTML 报告页面。"""
        if save_path is None:
            save_path = self.default_report_path(ticker)
        return write_report_tree(final_state, ticker, save_path, settings=self.run_settings(), html=html)

    def default_report_path(self, ticker) -> Path:
        """返回默认报告输出路径：位于 results_dir 下，并使用当前时间命名。"""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.config["results_dir"]) / "reports" / f"{safe_ticker_component(ticker)}_{stamp}"

    def create_run_state(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """构造运行初始状态；propagate() 和 CLI 都从这里开始。

        向所有 Agent 注入已解析的标的身份（#814）。Memory Log 经验不在这里
        加载，而由 Graph 的 Memory Log 节点与分析师并行结算并读取
        （见 ``_memory_step``）。"""
        return self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            instrument_context=self.resolve_instrument_context(company_name, asset_type, trade_date),
            portfolio_context=portfolio.render(company_name) if portfolio is not None else "",
        )

    def _memory_step(self, state):
        """Graph 的 Memory Log 节点与分析师并行运行（#1428）：
        结算全部 ticker 到期的历史决策（#1445），再向 Portfolio Manager
        返回在当前交易日期之前已经产生的经验（#1251）。

        结算需要获取价格，并请求模型对每笔决策生成反思，因此安排在分析师
        执行期间而非之前。此节点失败不会中断整次分析：保留分析师结果、
        继续使用日志中已有的经验，并在报告中说明哪些决策未能结算。"""
        note = ""
        try:
            # 若其他运行正在结算同一日志，则由其完成结算，当前运行继续执行。
            done = self.settle_all_pending(wait=False)
            if done.failed:
                note = (f"{len(done.failed)} past decision(s) could not be settled this run "
                        "and stay pending.")
        except Exception as exc:
            logger.warning("Settling past decisions failed: %s", exc)
            note = f"Past decisions could not be settled this run ({type(exc).__name__}); they stay pending."
        try:
            past_context = self.memory_log.get_past_context(
                state["company_of_interest"], as_of=self._memory_as_of(state["trade_date"]))
        except Exception as exc:
            logger.warning("Reading the memory log failed: %s", exc)
            past_context = ""
            note = note or f"The memory log could not be read this run ({type(exc).__name__})."
        return {"past_context": past_context, "memory_note": note}

    def settle_pending(self, company_name) -> settlement.Settlement:
        """结算当前 ticker 中持仓观察期已结束的决策。

        每次运行只会在分析师执行期间结算已到期的历史决策；本次新决策
        要等后续运行才到期。如果调用者已经完成该 ticker 的分析
        （例如回测批处理、定时任务），可以调用本方法立即尝试结算。
        返回成功结算和失败的项目。"""
        with run_config(self.config):
            return settlement.settle_pending(company_name, self.memory_log, self.reflector, self.config)

    def settle_all_pending(self, wait: bool = True) -> settlement.Settlement:
        """结算所有 ticker 中持仓观察期已结束的决策（#1445）。

        对轮换 ticker 的调度器而言，如果停止分析某个 ticker，
        其决策可能永远处于待结算状态，后续也无法使用相应经验。
        设置 ``wait=False`` 时，不等待同一日志上已在执行的结算任务。"""
        with run_config(self.config):
            return settlement.settle_all_pending(self.memory_log, self.reflector, self.config, wait=wait)

    def record_decision(self, company_name, trade_date, final_state):
        """记录已完成运行：保存状态日志，并将决策写入记忆日志，
        供下次分析相同 ticker 时反思。propagate() 和 CLI 都在此收尾。"""
        self._log_state(trade_date, final_state)
        decision = final_state.get("final_trade_decision")
        if not decision:
            logger.warning("No final decision for %s on %s; nothing added to the memory log",
                           company_name, trade_date)
            return
        self.memory_log.store_decision(
            ticker=company_name, trade_date=trade_date, final_trade_decision=decision,
            rating=run_rating(final_state),
        )

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock",
                   checkpoint_thread_id: str | None = None, portfolio=None):
        """执行 Graph，并将最终状态写入磁盘和记忆日志。"""
        init_agent_state = self.create_run_state(company_name, trade_date, asset_type, portfolio)
        args = self.propagator.get_graph_args()

        # 注入 checkpoint_scope 提供的 thread_id，使相同
        #  ticker、日期和图结构能够恢复；不同组合重新开始（#1089）。
        if checkpoint_thread_id is not None:
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_thread_id

        # 传入 None 表示恢复已有检查点；init_agent_state 表示全新运行（#1249）。
        graph_input = self.checkpoint_input(init_agent_state)
        if self.debug:
            # 每个状态都包含之前的消息，因此每条消息只打印一次（#1027）。
            final_state, printed = {}, set()
            for messages, state in self.stream_run(graph_input, **args):
                for msg in messages:
                    key = getattr(msg, "id", None) or (type(msg).__name__, getattr(msg, "content", None))
                    if key not in printed:
                        printed.add(key)
                        msg.pretty_print()
                if state is not None:
                    final_state.update(state)
        else:
            final_state = self.graph.invoke(graph_input, **args)

        self.record_decision(company_name, trade_date, final_state)

        # 成功完成后清理检查点，避免下次读取过期状态。
        self.clear_checkpoint_on_success(company_name, trade_date, asset_type, portfolio)

        return final_state, run_rating(final_state)

    def stream_run(self, graph_input, **args):
        """以 ``(messages, state)`` 的形式流式返回运行结果。

        ``messages`` 包含各 Agent（包括分析师）的消息；``state`` 为顶层
        步骤执行后的运行状态。若是分析师子图内部步骤，只有报告提交后
        才能取得该报告，否则为 None。

        每个分析师都在独立子图中运行；顶层状态须等待最慢的分析师完成，
        才统一接收所有报告。因此，为了及时输出消息和报告，应在各自
        子图任务完成时就流式返回其结果。"""
        args = {**args, "stream_mode": ["values", "tasks"]}
        # 与 propagate() 一样，每次工具调用使用该 Graph 自身的配置，
        # 即使进程全局配置后来改变也不受影响。每个步骤在本次运行的
        # 上下文中执行，调用者在步骤之间也能保持自己的配置。
        context = run_config_context(self.config)
        stream = context.run(self.graph.stream, graph_input, subgraphs=True, **args)
        try:
            while (step := context.run(next, stream, None)) is not None:
                namespace, mode, chunk = step
                if namespace:
                    result = chunk.get("result") if mode == "tasks" else None
                    if isinstance(result, dict):
                        report = {k: v for k, v in result.items() if k != "messages" and v}
                        if result.get("messages") or report:
                            yield result.get("messages", []), report or None
                elif mode == "values":
                    yield chunk.get("messages", []), chunk
        finally:
            context.run(stream.close)

    def _log_state(self, trade_date, final_state):
        """将运行最终状态保存为 JSON，目录按本次 ticker 划分。"""
        entry = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
            },
            "trader_investment_plan": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
            "final_rating": run_rating(final_state),
            "run_settings": self.run_settings(),
        }

        # 拒绝会造成结果目录路径逃逸的 ticker。
        safe_ticker = safe_ticker_component(final_state["company_of_interest"])
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            # 报告可能采用任意语言，且此文件供人直接阅读。
            json.dump(entry, f, indent=4, ensure_ascii=False)
