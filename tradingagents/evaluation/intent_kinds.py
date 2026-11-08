"""Exact agent/schema pairs that emit structured model output via function calling.

These are model output contracts, not data tools executed by LangGraph ToolNode.
An arbitrary unknown function name remains an unauthorized business-tool request.
"""

from pydantic import BaseModel

from tradingagents.agents.schemas import (
    PortfolioDecision,
    ResearchPlan,
    SentimentReport,
    TraderProposal,
)

STRUCTURED_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "Research Manager": ResearchPlan,
    "Trader": TraderProposal,
    "Portfolio Manager": PortfolioDecision,
    "Sentiment Analyst": SentimentReport,
}


def structured_output_schema(scope: str | None, name: str | None) -> type[BaseModel] | None:
    """Only recognize an exact schema name in its owning Graph agent's scope."""
    if not scope or not name:
        return None
    agent_name = scope.split(":", 1)[0]
    schema = STRUCTURED_OUTPUT_SCHEMAS.get(agent_name)
    return schema if schema is not None and name == schema.__name__ else None


def is_structured_output(scope: str | None, name: str | None) -> bool:
    return structured_output_schema(scope, name) is not None
