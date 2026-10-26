from typing import Any

from tradingagents.agents.state import InvestDebateState, RiskDebateState


class Propagator:
    """负责图中状态的初始化与传播。"""

    def __init__(self, max_recur_limit=100):
        """使用配置参数初始化。"""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        portfolio_context: str = "",
    ) -> dict[str, Any]:
        """创建 Agent Graph 的初始状态。
        
        ``instrument_context`` 是运行开始时解析一次的确定性标的身份信息
        （见 ``TradingAgentsGraph.resolve_instrument_context``）。
        若该值为空，Agent 将通过 ``get_instrument_context_from_state``
        退回到仅使用 ticker 的上下文。"""
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "memory_note": "",
            "portfolio_context": portfolio_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """生成调用 Graph 时使用的参数。
        
        参数：
            callbacks：用于跟踪工具执行的可选回调列表。
                注意：LLM 回调在 LLM 构造函数中单独处理。"""
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
