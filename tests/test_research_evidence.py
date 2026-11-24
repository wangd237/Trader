"""Evidence summary provenance, fallback and downstream prompt regressions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.analysts.turn import sanitize_analyst_report
from tradingagents.agents.evidence import (
    EvidenceExcerpt,
    ResearchEvidenceSummary,
    _render_verified,
    create_research_evidence_node,
    get_research_context_from_state,
)
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator


REPORTS = {
    "market": "Price 344 HKD on 2026-11-20, RSI 26.79. Market trend weak." + " Chart details." * 20,
    "fundamentals": "H1 revenue RMB 3964 billion? Unit UNVERIFIED. Q3 figures unavailable." + " Filing detail." * 20,
    "news": "Q3 results announced 2026-11-18. Full PDF was not retrieved." + " Announcement detail." * 20,
    "sentiment": "Social feeds unavailable; confidence low, not neutral evidence." + " Coverage detail." * 20,
}


def _state(**updates):
    state = {
        "company_of_interest": "0700.HK",
        "trade_date": "2026-11-20",
        "market_report": REPORTS["market"],
        "fundamentals_report": REPORTS["fundamentals"],
        "news_report": REPORTS["news"],
        "sentiment_report": REPORTS["sentiment"],
        "trader_investment_plan": "Trader proposal",
        "investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": "", "count": 0,
        },
        "risk_debate_state": {
            "history": "", "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "current_aggressive_response": "",
            "current_conservative_response": "", "current_neutral_response": "",
            "latest_speaker": "", "count": 0,
        },
    }
    return {**state, **updates}


def _excerpt(source, kind="metric", quote=None):
    report = REPORTS[source]
    return EvidenceExcerpt(source=source, kind=kind, quote=quote or report[:42])


def _model_with_structured_result(result):
    structured = MagicMock()
    structured.invoke.return_value = result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
def test_summary_verifies_exact_source_quotes_and_is_compact():
    examples = ResearchEvidenceSummary(excerpts=[
        _excerpt("market"),
        _excerpt("fundamentals"),
        _excerpt("news", "event"),
        _excerpt("sentiment", "gap"),
    ])
    text = _render_verified(examples, REPORTS)
    assert "market@0" in text
    assert "not neutral evidence" not in text  # the unselected tail remains outside the summary
    assert len(text) < sum(map(len, REPORTS.values()))
    assert "Source" not in text  # no synthetic market claims


@pytest.mark.unit
def test_unknown_quote_fails_closed_to_full_reports():
    bad = ResearchEvidenceSummary(excerpts=[
        EvidenceExcerpt(source="market", kind="metric", quote="Price is 999 HKD on 2026-11-20."),
        _excerpt("fundamentals"),
        _excerpt("news"),
        _excerpt("sentiment"),
    ])
    node = create_research_evidence_node(_model_with_structured_result(bad))
    result = node(_state())
    assert result["research_evidence_status"] == "fallback"
    state = {**_state(), **result}
    assert "Price 344 HKD" in get_research_context_from_state(state)
    assert "Q3 figures unavailable" in get_research_context_from_state(state)


@pytest.mark.unit
def test_short_report_coverage_required_for_each_present_source():
    summary = ResearchEvidenceSummary(excerpts=[_excerpt("market")])
    with pytest.raises(ValueError, match="coverage"):
        _render_verified(summary, REPORTS)


@pytest.mark.unit
def test_missing_report_is_explicit_in_valid_summary():
    reports = {"market": REPORTS["market"]}
    summary = ResearchEvidenceSummary(excerpts=[_excerpt("market")])
    rendered = _render_verified(summary, reports)
    assert "[fundamentals | missing]" in rendered


@pytest.mark.unit
def test_valid_summary_is_shared_by_bull_and_risk_without_raw_reports():
    excerpts = ResearchEvidenceSummary(excerpts=[
        _excerpt("market"), _excerpt("fundamentals"),
        _excerpt("news", "event"), _excerpt("sentiment", "gap"),
    ])
    model = _model_with_structured_result(excerpts)
    result = create_research_evidence_node(model)(_state())
    assert result["research_evidence_status"] == "ready"
    model.with_structured_output.return_value.invoke.assert_called_once()
    state = _state(**result)

    captured = []
    debater = MagicMock()
    debater.invoke.side_effect = lambda prompt: (
        captured.append(prompt) or MagicMock(content="An argument")
    )
    create_bull_researcher(debater)(state)
    create_aggressive_debator(debater)(state)
    assert len(captured) == 2
    for prompt in captured:
        assert "[market@0" in prompt
        assert "Market trend weak." not in prompt  # non-selected raw material stays outside
        assert "Trader proposal" in prompt or "Bull" in prompt


@pytest.mark.unit
def test_explicit_decision_labels_removed_but_research_evidence_kept():
    report = (
        "Price 344 HKD, RSI 26.79.\n"
        "**Rating**: Sell\n"
        "FINAL TRANSACTION PROPOSAL: **SELL**\n"
        "2026-11-18 company results announced.\n"
        "交易建议：减仓\n"
    )
    cleaned = sanitize_analyst_report(report)
    assert "Price 344 HKD" in cleaned
    assert "2026-11-18" in cleaned
    assert "Rating" not in cleaned
    assert "TRANSACTION PROPOSAL" not in cleaned
    assert "减仓" not in cleaned
