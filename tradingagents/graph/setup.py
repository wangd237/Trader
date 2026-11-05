import logging
from collections import Counter
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.analysts.turn import (
    BLOCKED_REPEAT_RESULT,
    TOOL_LOOP_WARNING,
    WRAP_UP,
    WRAP_UP_REPEATED,
)
from tradingagents.agents.state import AgentState
from tradingagents.dataflows.date_window import is_historical

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic
from .tool_loop_guard import (
    cached_historical_tool_messages,
    consecutive_identical_tool_rounds,
    repeats_warned_tool_request,
)

logger = logging.getLogger(__name__)

# 共享条件路由可能返回的全部目标节点。所有由该路由控制的边
# 都映射完整目标集合，避免 Prompt、国际化或重构造成发言人名称变化时，
# 兜底返回值缺少 path_map 映射而使 LangGraph 在运行中崩溃（#1088）。
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


def _tools_or_done(state) -> str:
    """决定分析师当前轮次的去向：继续调用工具，或生成报告并结束。"""
    return "tools" if state["messages"][-1].tool_calls else END


def _analyst_graph(
    spec, agent, max_tool_rounds: int,
    max_stagnant_tool_rounds: int = 3,
    reuse_historical_tool_results: bool = True,
):
    """每名分析师使用独立子图执行模型和工具调用，并维护私有消息历史。

    子图只返回该分析师的报告，因此并行分析师不会写入同一个状态键，
    其工具调用记录也不会混入其他分析师的消息。达到 ``max_tool_rounds``
    后，会要求模型给出最终报告，并在这一轮结束，不管模型如何回答；
    这可防止持续调用工具的模型触发图的递归上限（#1420）。"""
    output = TypedDict(f"{spec.key.capitalize()}Report", {spec.report_key: str})
    graph = StateGraph(AgentState, output_schema=output)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    if not spec.tools:
        graph.add_edge("agent", END)
        return graph.compile()

    def calls(messages):
        return [call["name"] for m in messages for call in (getattr(m, "tool_calls", None) or [])]

    def rounds(messages) -> int:
        return sum(1 for m in messages if getattr(m, "tool_calls", None))

    def warned(messages) -> bool:
        return any(
            isinstance(message, HumanMessage)
            and isinstance(message.content, str)
            and message.content.startswith(TOOL_LOOP_WARNING)
            for message in messages
        )

    def more_or_wrap_up(state) -> str:
        messages = state["messages"]
        if rounds(messages) >= max_tool_rounds:
            return "wrap_up"
        if (
            max_stagnant_tool_rounds > 0
            and consecutive_identical_tool_rounds(messages) >= max_stagnant_tool_rounds
        ):
            if warned(messages):
                logger.warning("%s still repeats identical tool results after one warning; wrapping up",
                               spec.agent_node)
                return "wrap_up"
            logger.warning("%s repeated identical tool results %d times; issuing one warning",
                           spec.agent_node, max_stagnant_tool_rounds)
            return "warn"
        return "agent"

    def latest_requested_tools(messages):
        return next(
            (m.tool_calls for m in reversed(messages) if getattr(m, "tool_calls", None)), []
        )

    def warn(state):
        names = ", ".join(sorted({call["name"] for call in latest_requested_tools(state["messages"])}))
        message = (
            f"{TOOL_LOOP_WARNING} Repeated tool(s): {names}. "
            "Do not call these tools again with the same arguments unless you have new evidence "
            "that warrants refreshing them. Use the results already provided, choose a "
            "different useful tool or strategy, or write your final report. "
            "This is your one opportunity to adjust before repeated requests are blocked."
        )
        return {"messages": [HumanMessage(message)]}

    def cached_results(state):
        return cached_historical_tool_messages(
            state["messages"], historical_run=is_historical(state["trade_date"]),
        ) if reuse_historical_tool_results else None

    def tools_or_done(state):
        if _tools_or_done(state) == END:
            return END
        if max_stagnant_tool_rounds > 0 and repeats_warned_tool_request(
            state["messages"], TOOL_LOOP_WARNING
        ):
            return "blocked_tools"
        return "cached_tools" if cached_results(state) is not None else "tools"

    def blocked_tools(state):
        pending = state["messages"][-1]
        logger.warning("%s blocked an unchanged tool request after the corrective warning",
                       spec.agent_node)
        return {
            "messages": [
                ToolMessage(
                    content=BLOCKED_REPEAT_RESULT,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
                for call in pending.tool_calls
            ]
        }

    def cached_tools(state):
        results = cached_results(state)
        if results is None:
            raise ValueError("cached tool round is no longer reusable")
        logger.info("%s reused %d historical tool result(s)", spec.agent_node, len(results))
        return {"messages": results}

    def wrap_up(state):
        counts = ", ".join(f"{name} x{n}" for name, n in Counter(calls(state["messages"])).most_common())
        stagnant_rounds = consecutive_identical_tool_rounds(state["messages"])
        blocked = (
            isinstance(state["messages"][-1], ToolMessage)
            and state["messages"][-1].content == BLOCKED_REPEAT_RESULT
        )
        if blocked or (
            max_stagnant_tool_rounds > 0
            and stagnant_rounds >= max_stagnant_tool_rounds
            and warned(state["messages"])
        ):
            latest_calls = latest_requested_tools(state["messages"])
            names = ", ".join(sorted({call["name"] for call in latest_calls}))
            message = (
                f"{WRAP_UP_REPEATED} Repeated tool(s): {names}. "
                "You have already received one strategy-correction warning. "
                "Write the report using available evidence. Explain missing or uncertain "
                "data rather than guessing; do not request further tools."
            )
            logger.warning("%s stopped after an unsuccessful correction (%s)",
                           spec.agent_node, names)
        else:
            message = WRAP_UP
            logger.warning("%s used its %d tool rounds (%s); asking for its report",
                           spec.agent_node, rounds(state["messages"]), counts)
        return agent({**state, "messages": [*state["messages"], HumanMessage(message)]})

    graph.add_node("tools", ToolNode(list(spec.tools)))
    graph.add_node("cached_tools", cached_tools)
    graph.add_node("blocked_tools", blocked_tools)
    graph.add_node("warn", warn)
    graph.add_node("wrap_up", wrap_up)
    graph.add_conditional_edges(
        "agent", tools_or_done, ["tools", "cached_tools", "blocked_tools", END]
    )
    graph.add_conditional_edges("tools", more_or_wrap_up, ["agent", "warn", "wrap_up"])
    graph.add_conditional_edges("cached_tools", more_or_wrap_up, ["agent", "warn", "wrap_up"])
    graph.add_edge("blocked_tools", "wrap_up")
    graph.add_edge("warn", "agent")
    graph.add_edge("wrap_up", END)
    return graph.compile()


class GraphSetup:
    """负责 Agent Graph 的搭建与配置。"""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        conditional_logic: ConditionalLogic,
        max_tool_rounds: int,
        max_stagnant_tool_rounds: int = 3,
        reuse_historical_tool_results: bool = True,
    ):
        """使用必需的组件进行初始化。"""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.conditional_logic = conditional_logic
        self.max_tool_rounds = max_tool_rounds
        if max_stagnant_tool_rounds < 0:
            raise ValueError("max_stagnant_tool_rounds must be >= 0")
        self.max_stagnant_tool_rounds = max_stagnant_tool_rounds
        self.reuse_historical_tool_results = reuse_historical_tool_results

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals"), memory_node=None
    ):
        """构建并编译 Agent 工作流图。

        参数：
            selected_analysts (list)：需要启用的分析师类型，可选：
                - "market"：市场分析师
                - "social"：情绪分析师
                - "news"：新闻分析师
                - "fundamentals"：基本面分析师
            memory_node：结算历史决策并返回本次运行 ``past_context`` 的节点；
                与分析师节点并行执行。"""

        # 1. 确定启用哪些分析师
        plan = build_analyst_execution_plan(selected_analysts)

        # 2.创建各个分析师的agent函数
        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # 3. 创建主图
        workflow = StateGraph(AgentState)
        # 4. 将每个分析师的子图添加到主图中
        for spec in plan.specs:
            workflow.add_node(spec.agent_node,
                              _analyst_graph(spec, analyst_factories[spec.key](), self.max_tool_rounds,
                                             self.max_stagnant_tool_rounds,
                                             self.reuse_historical_tool_results))

        # 5. 将研究团队和交易团队的节点添加到主图中
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # 分析师并行执行；只有全部分析师提交报告后才开始研究辩论。
        # Memory Log 也同时结算历史决策，但只有 Portfolio Manager 读取经验。
        first_steps = [spec.agent_node for spec in plan.specs]
        # 6.并行启动分析师和memory log
        if memory_node is not None:
            workflow.add_node("Memory Log", memory_node)
            first_steps.append("Memory Log")
        for node in first_steps:
            workflow.add_edge(START, node)
        workflow.add_edge(first_steps, "Bull Researcher")

        # 研究辩论的两条边共用完整的 DEBATE_PATH_MAP（#1088）。
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # 三条风险分析边共用完整的 RISK_ANALYSIS_PATH_MAP（#1088）。
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
