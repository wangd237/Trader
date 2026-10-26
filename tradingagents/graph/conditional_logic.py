from tradingagents.agents.state import AgentState


class ConditionalLogic:
    """封装控制 Graph 执行流程的条件判断逻辑。"""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """根据配置参数进行初始化。"""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_debate(self, state: AgentState) -> str:
        """判断多空研究辩论是否继续。"""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 多方和空方各进行 max_debate_rounds 轮发言
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """判断风险分析讨论是否继续。"""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 三名风险分析师各进行 max_risk_discuss_rounds 轮发言
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
