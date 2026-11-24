"""Regression: debate prompts should not repeat latest arguments already in history.

P0 only changes prompt assembly. State keys and complete debate transcripts stay intact.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.context import opponent_argument_for_prompt
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator


def _llm(recorded: dict):
    fake = MagicMock()
    fake.invoke.side_effect = lambda prompt: (
        recorded.__setitem__("prompt", prompt) or MagicMock(content="new argument")
    )
    return fake


def _base_state() -> dict:
    return {
        "company_of_interest": "0700.HK",
        "asset_type": "stock",
        "market_report": "market evidence",
        "sentiment_report": "sentiment evidence",
        "news_report": "news evidence",
        "fundamentals_report": "fundamentals evidence",
        "trader_investment_plan": "Trader plan",
    }


@pytest.mark.unit
def test_opponent_context_avoids_duplication_but_keeps_fallbacks():
    latest = "Bull Analyst: unique-bull-point"
    history = "Older point\n" + latest
    assert opponent_argument_for_prompt(history, latest, "bull").startswith("(already included")
    assert latest not in opponent_argument_for_prompt(history, latest, "bull")
    assert opponent_argument_for_prompt("Older point", latest, "bull") == latest
    assert "has not spoken yet" in opponent_argument_for_prompt("", "", "bull")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "latest"),
    [
        (create_bull_researcher, "Bear Analyst: DISTINCT_BEAR_ARGUMENT"),
        (create_bear_researcher, "Bull Analyst: DISTINCT_BULL_ARGUMENT"),
    ],
)
def test_research_prompt_uses_each_opponent_argument_once(factory, latest):
    state = _base_state()
    state["investment_debate_state"] = {
        "history": "prior round\n" + latest,
        "bull_history": "",
        "bear_history": "",
        "current_response": latest,
        "count": 1,
    }
    recorded = {}
    updated = factory(_llm(recorded))(state)["investment_debate_state"]

    assert recorded["prompt"].count(latest) == 1
    assert "(already included in the debate history above)" in recorded["prompt"]
    assert updated["history"].endswith("new argument")
    assert updated["count"] == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [create_aggressive_debator, create_conservative_debator, create_neutral_debator],
)
def test_risk_prompt_does_not_duplicate_any_existing_opponent_argument(factory):
    arguments = {
        "current_aggressive_response": "Aggressive Analyst: UNIQUE_AGGRESSIVE_POINT",
        "current_conservative_response": "Conservative Analyst: UNIQUE_CONSERVATIVE_POINT",
        "current_neutral_response": "Neutral Analyst: UNIQUE_NEUTRAL_POINT",
    }
    history = "\n".join(arguments.values())
    state = _base_state()
    state["risk_debate_state"] = {
        **arguments,
        "history": history,
        "aggressive_history": arguments["current_aggressive_response"],
        "conservative_history": arguments["current_conservative_response"],
        "neutral_history": arguments["current_neutral_response"],
        "latest_speaker": "Neutral",
        "count": 3,
    }
    recorded = {}
    updated = factory(_llm(recorded))(state)["risk_debate_state"]

    for argument in arguments.values():
        assert recorded["prompt"].count(argument) == 1
    assert updated["history"].startswith(history)
    assert updated["count"] == 4


@pytest.mark.unit
def test_risk_prompt_still_includes_argument_missing_from_history():
    state = _base_state()
    state["risk_debate_state"] = {
        "history": "Aggressive Analyst: earlier argument",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "current_aggressive_response": "Aggressive Analyst: earlier argument",
        "current_conservative_response": "Conservative Analyst: NEWLY_RESTORED_POINT",
        "current_neutral_response": "",
        "latest_speaker": "Aggressive",
        "count": 1,
    }
    recorded = {}
    create_aggressive_debator(_llm(recorded))(state)
    assert recorded["prompt"].count("Conservative Analyst: NEWLY_RESTORED_POINT") == 1
    assert "neutral analyst has not spoken yet" in recorded["prompt"]
