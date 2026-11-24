"""Evidence ID selection, exact provenance, partial tolerance and fallback tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.analysts.turn import sanitize_analyst_report
from tradingagents.agents.evidence import (
    EvidenceReference,
    ResearchEvidenceSelection,
    _build_evidence_catalog,
    _render_verified,
    create_research_evidence_node,
    get_research_context_from_state,
)
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator

REPORTS = {
    "market": "Price 344 HKD on 2026-11-20, RSI 26.79. Market trend weak." + " Chart details." * 48,
    "fundamentals": "H1 revenue RMB 3964 billion? Unit UNVERIFIED. Q3 figures unavailable."
    + " Filing detail." * 20,
    "news": "Q3 results announced 2026-11-18. Full PDF was not retrieved."
    + " Announcement detail." * 20,
    "sentiment": "Social feeds unavailable; confidence low, not neutral evidence."
    + " Coverage detail." * 20,
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


def _select_all(reports=REPORTS, *, kinds=None):
    catalog = _build_evidence_catalog(reports)
    refs = []
    for source in reports:
        ref = next(r for r, span in catalog.items() if span.source == source)
        refs.append(EvidenceReference(ref=ref, kind=(kinds or {}).get(source, "metric")))
    return ResearchEvidenceSelection(excerpts=refs)


def _model_with_structured_result(result):
    structured = MagicMock()
    structured.invoke.return_value = result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
def test_ids_resolve_to_exact_source_spans_and_compress():
    selection = _select_all()
    catalog = _build_evidence_catalog(REPORTS)
    summary, rejected = _render_verified(selection, catalog, REPORTS)
    assert rejected == []
    assert "[market@0 | M001 | metric]" in summary
    assert "Price 344 HKD" in summary
    assert summary.count("Market trend weak.") == 1
    assert len(summary) < sum(map(len, REPORTS.values()))
    for ref in selection.excerpts:
        span = catalog[ref.ref]
        assert span.text == REPORTS[span.source][span.offset:span.offset + len(span.text)]


@pytest.mark.unit
def test_invalid_reference_cannot_invent_numbers_or_break_other_valid_evidence():
    selection = _select_all()
    selection.excerpts.insert(0, EvidenceReference(ref="M999", kind="metric"))
    node = create_research_evidence_node(_model_with_structured_result(selection))
    result = node(_state())
    assert result["research_evidence_status"] == "ready"
    assert result["research_evidence_error"] == ""
    assert "999 HKD" not in result["research_evidence"]
    assert "Price 344 HKD" in result["research_evidence"]


@pytest.mark.unit
def test_invalid_reference_that_removes_source_coverage_falls_back():
    selection = _select_all()
    selection.excerpts[0] = EvidenceReference(ref="M999", kind="metric")
    node = create_research_evidence_node(_model_with_structured_result(selection))
    result = node(_state())
    assert result["research_evidence_status"] == "fallback"
    assert "insufficient_coverage:market" in result["research_evidence_error"]
    context = get_research_context_from_state(_state(**result))
    assert REPORTS["market"] in context
    assert REPORTS["fundamentals"] in context


@pytest.mark.unit
def test_full_original_is_kept_when_structured_call_fails():
    model = _model_with_structured_result(None)
    model.with_structured_output.return_value.invoke.side_effect = RuntimeError("bad JSON")
    result = create_research_evidence_node(model)(_state())
    assert result["research_evidence_status"] == "fallback"
    assert result["research_evidence_error"] == "structured_output_error:RuntimeError"
    assert REPORTS["news"] in get_research_context_from_state(_state(**result))


@pytest.mark.unit
def test_missing_report_is_labelled_and_does_not_require_coverage():
    reports = {"market": REPORTS["market"]}
    selection = _select_all(reports)
    rendered, rejected = _render_verified(selection, _build_evidence_catalog(reports), reports)
    assert not rejected
    assert "[fundamentals | missing]" in rendered


@pytest.mark.unit
def test_verbatim_markdown_survives_original_fallback_incident():
    """2026-11-24 trace: a misplaced ** in a quoted cash-flow claim caused fallback.

    The selector now returns F001, not the retyped sentence; the exact original
    with its original Markdown formatting is copied by deterministic code.
    """
    original = (
        "1. **2Q 经营现金流骤降至 52.710 十亿元（环比 −48%），而 2Q 净利润为 "
        "56.022 十亿元，OCF/净利润仅 0.94x，首次低于 1.0x。**\n\n"
        + "已核实的原始数据。" * 60
    )
    reports = {"fundamentals": original}
    catalog = _build_evidence_catalog(reports)
    ref = next(key for key, span in catalog.items() if "52.710" in span.text)
    selection = ResearchEvidenceSelection(
        excerpts=[EvidenceReference(ref=ref, kind="negative")]
    )
    summary, rejected = _render_verified(selection, catalog, reports)
    assert not rejected
    assert "**2Q 经营现金流骤降至 52.710" in summary
    assert "首次低于 1.0x。**" in summary
    assert "**，而 2Q" not in summary
    assert "[fundamentals@" in summary


@pytest.mark.unit
def test_large_reports_need_two_distinct_segments_for_coverage():
    reports = {"market": "Market valuation downside risk. " * 55}
    catalog = _build_evidence_catalog(reports)
    selection = ResearchEvidenceSelection(excerpts=[
        EvidenceReference(ref=next(iter(catalog)), kind="negative")
    ])
    with pytest.raises(ValueError, match="insufficient_coverage:market"):
        _render_verified(selection, catalog, reports)


@pytest.mark.unit
def test_duplicate_id_is_skipped_instead_of_discarding_all_evidence():
    selection = _select_all()
    selection.excerpts.append(selection.excerpts[0])
    text, rejected = _render_verified(
        selection, _build_evidence_catalog(REPORTS), REPORTS
    )
    assert rejected == ["M001"]
    assert text.count("[market@") == 1


@pytest.mark.unit
def test_oversized_candidate_input_does_not_call_model(monkeypatch):
    from tradingagents.agents import evidence

    monkeypatch.setattr(evidence, "MAX_PROMPT_CHARS", 10)
    llm = _model_with_structured_result(_select_all())
    result = create_research_evidence_node(llm)(_state())
    assert result["research_evidence_status"] == "fallback"
    assert "candidate_prompt_too_long" in result["research_evidence_error"]
    llm.with_structured_output.return_value.invoke.assert_not_called()


@pytest.mark.unit
def test_evidence_is_shared_by_bull_and_risk_without_raw_reports():
    selection = _select_all(kinds={"news": "event", "sentiment": "gap"})
    llm = _model_with_structured_result(selection)
    result = create_research_evidence_node(llm)(_state())
    assert result["research_evidence_status"] == "ready"
    llm.with_structured_output.return_value.invoke.assert_called_once()

    captured = []
    debater = MagicMock()
    debater.invoke.side_effect = lambda prompt: (
        captured.append(prompt) or MagicMock(content="An argument")
    )
    state = _state(**result)
    create_bull_researcher(debater)(state)
    create_aggressive_debator(debater)(state)

    assert len(captured) == 2
    for prompt in captured:
        assert "[market@0 | M001" in prompt
        assert REPORTS["market"] not in prompt
    assert "Trader proposal" not in captured[0]
    assert "Trader proposal" in captured[1]


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
