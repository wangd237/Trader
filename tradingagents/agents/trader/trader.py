"""Trader：将 Research Manager 的投资计划转换成具体交易提案。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # 研究计划汇总了辩论，却可能缺少具体价格依据。
        # 因此向 Trader 补充技术分析报告，以实际 ATR、支撑阻力位及当前价格
        # 指导入场和止损价（#1167）。若未选市场分析师，就不注入空报告。
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    + grounding
                    # 入场和止损字段必须是具体价格，不能是百分比（例如 15%）；
                    # 后者会导致结构化解析失败（#1288）。
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; convert a percentage distance to the price level it "
                    "implies, or omit the field if you cannot state a number. "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    "Make an informed, strategic trading decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Hold / Sell. A research "
                    "recommendation of Overweight is a Buy and Underweight is a Sell, "
                    "sized by how strong the case is; conflict alone is not a Hold.\n"
                    "- **Reasoning**: why, against the plan and the price structure\n"
                    "- **Entry Price**, **Stop Loss**, **Position Sizing**: when you can state them"
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
        }

    return trader_node
