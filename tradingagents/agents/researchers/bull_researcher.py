from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bear analyst"
        )
        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )
        prompt = (
    f"""你是一名看多分析师（多头），主张投资 {target_label}。你的任务是建立一个有力的、基于充分证据的论点，重点强调其增长潜力、竞争优势和积极的市场指标。请充分利用所提供的研究和数据，有效化解疑虑并有力反驳看跌（空头）观点。

重点关注以下维度：
- 增长潜力：突出该公司的市场机遇、营收预期以及业务的可扩展性/规模化能力。
- 竞争优势：强调独特的产品壁垒、强大的品牌影响力或主导性的市场地位等因素。
- 积极指标：以财务健康状况、行业发展趋势以及近期的正面利好新闻作为有力佐证。
- 反驳空头论点：结合具体数据和严密逻辑对空方观点进行批判性分析，全面回应其担忧，并证明为何多方立场更具说服力与优势。
- 互动与辩论感：采用对话风格陈述你的论据，直接切入并针对空头分析师的观点展开有效辩论，而非仅仅罗列枯燥的数据。

可用资源：
{instrument_context}
市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新全球时事新闻：{news_report}
{fundamentals_label}：{fundamentals_report}
辩论对话历史记录：{history}
空方最新论点：{current_response}

请利用上述信息输出一份具有说服力的多方论点，反驳空方的担忧，并展开一场针锋相对的动态辩论，以展现多方立场的强大优势。
"""
    + get_language_instruction()
)

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
