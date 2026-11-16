"""Offline regression coverage for extracted experiences and decision-time retrieval."""

from unittest.mock import MagicMock

import pytest

from tradingagents.memory.experience import ExperienceStore, build_experience_query, format_experiences
from tradingagents.memory.reflection import ExtractedExperience, ReflectionResult, Reflector
from tradingagents.memory import TradingMemoryLog, settlement
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating


def _lesson(**overrides):
    data = {
        "valuable": True,
        "scenario": "Reduce downside risk when cash flow weakens",
        "lesson": "Separate absolute risk reduction from relative alpha claims",
        "boundary": "A five-day window cannot prove a three-month thesis",
        "tags": ["risk_control", "relative_return", "time_horizon"],
    }
    data.update(overrides)
    return ExtractedExperience(**data)


@pytest.mark.unit
def test_single_structured_reflection_returns_prose_and_experience():
    llm = MagicMock()
    structured = llm.with_structured_output.return_value
    structured.invoke.return_value = ReflectionResult(reflection="Alpha was positive, but horizon too short.",
                                                       experience=_lesson())
    output = Reflector(llm).reflect_on_final_decision(
        "Rating: Underweight", -0.031, 0.032, include_experience=True
    )
    assert isinstance(output, ReflectionResult)
    assert output.experience.valuable
    assert structured.invoke.call_count == 1
    llm.invoke.assert_not_called()


@pytest.mark.unit
def test_reflection_free_text_fallback_keeps_log_only():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError()
    llm.invoke.return_value.content = "Five days are insufficient evidence."
    output = Reflector(llm).reflect_on_final_decision(
        "Rating: Hold", 0.01, 0.00, include_experience=True
    )
    assert output.reflection
    assert not output.experience.valuable
    assert llm.invoke.call_count == 1


@pytest.mark.unit
def test_store_idempotent_and_cross_ticker_retrieval(tmp_path):
    store = ExperienceStore({"memory_log_path": str(tmp_path / "trading_memory.md")})
    assert store.store(ticker="000001.SZ", trade_date="2026-09-30",
                       resolved_at="2026-10-15", experience=_lesson())
    assert not store.store(ticker="000001.SZ", trade_date="2026-09-30",
                           resolved_at="2026-10-15", experience=_lesson())
    assert len(store.load_entries()) == 1
    found = store.retrieve("9988.HK: cash flow weakens, reduce downside risk and evaluate alpha",
                           as_of="2026-10-15")
    assert len(found) == 1
    assert found[0]["ticker"] == "000001.SZ"
    assert "Separate absolute risk" in format_experiences(found)
    assert not store.retrieve("reduce downside risk", as_of="2026-10-14")
    assert not store.retrieve("completely unrelated satellite orbit")


@pytest.mark.unit
def test_nonvaluable_experience_is_not_stored(tmp_path):
    store = ExperienceStore({"memory_log_path": str(tmp_path / "trading_memory.md")})
    assert not store.store(ticker="AAPL", trade_date="2026-01-01",
                           resolved_at="2026-01-09",
                           experience=_lesson(valuable=False))
    assert not store.load_entries()


@pytest.mark.unit
def test_query_only_uses_relevant_state_fields():
    state = {
        "investment_plan": "cash flow decline",
        "trader_investment_plan": "reduce exposure",
        "risk_debate_state": {"history": "uncertain alpha"},
        "messages": ["MUST NOT BE INDEXED"],
    }
    query = build_experience_query(state)
    assert "cash flow" in query
    assert "reduce exposure" in query
    assert "uncertain alpha" in query
    assert "MUST NOT" not in query


@pytest.mark.unit
def test_settlement_writes_to_both_stores_once(tmp_path, monkeypatch):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "trading_memory.md")})
    store = ExperienceStore({"memory_log_path": str(tmp_path / "trading_memory.md")})
    log.store_decision("000001.SZ", "2026-09-30", "Rating: Underweight")
    reflector = MagicMock()
    reflector.reflect_on_final_decision.return_value = ReflectionResult(
        reflection="Alpha positive but observation is short.", experience=_lesson()
    )
    monkeypatch.setattr(settlement, "fetch_returns",
                        lambda *a, **kw: (-0.031, 0.032, 5, "2026-10-15"))
    config = {"_experience_store": store, "benchmark_map": {".SZ": "399001.SZ"}}
    assert len(settlement.settle_pending("000001.SZ", log, reflector, config).settled) == 1
    assert len(settlement.settle_pending("000001.SZ", log, reflector, config).settled) == 0
    assert log.load_entries()[0]["reflection"] == "Alpha positive but observation is short."
    assert len(store.load_entries()) == 1


@pytest.mark.unit
def test_pm_retrieves_based_on_current_risk_context(tmp_path):
    store = ExperienceStore({"memory_log_path": str(tmp_path / "trading_memory.md")})
    store.store(ticker="000001.SZ", trade_date="2026-09-30",
                resolved_at="2026-10-15", experience=_lesson())
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = PortfolioDecision(
        rating=PortfolioRating.UNDERWEIGHT,
        executive_summary="Lower exposure", investment_thesis="Cash flow weakness",
    )
    state = {
        "trade_date": "2026-11-16",
        "company_of_interest": "9988.HK",
        "risk_debate_state": {
            "history": "Reduce downside risk as cash flow weakens; distinguish alpha.",
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "current_aggressive_response": "",
            "current_conservative_response": "", "current_neutral_response": "",
            "count": 1,
        },
        "investment_plan": "Cash flow is weakening.",
        "trader_investment_plan": "Reduce exposure.",
    }
    pm = create_portfolio_manager(llm, experience_store=store, memory_as_of=lambda _: "2026-11-16")
    pm(state)
    prompt = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert "Separate absolute risk reduction" in prompt
    assert "000001.SZ" in prompt
    assert "historical" not in prompt.lower() or "fallible" in prompt.lower()
