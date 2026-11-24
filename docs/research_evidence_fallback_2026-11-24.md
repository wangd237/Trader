# Research Evidence fallback incident (2026-11-24)

## Reproduction and observations

- Analyzed: `0700.HK`, as-of `2026-11-20`; run on `feat/research-evidence-summary`.
- LangSmith trace ID: `01a20b82-264d-7687-aa27-196beb80d3a2`; the local, privately exported `research_evidence.json` is **not** committed.
- Four complete analyst reports: **39,676 characters** (market 7,429; fundamentals 17,197; news 10,628; sentiment 4,422).
- The Research Evidence LLM call consumed **24,323 input + 18,456 output = 42,779 tokens** (reasoning accounted for 16,539 output tokens).
- Pydantic successfully parsed 28 structured evidence excerpts. **27/28** were exact substrings of their declared reports.
- Excerpt #13 was a cash-flow claim from Fundamentals: the numbers, dates and meaning were unchanged, but the model moved the Markdown bold closing delimiter. The previous `original.find(quote)` returned `-1`, invalidating the **entire** summary. The node returned `research_evidence_status=fallback` and downstream agents received all original reports.

## Root cause

The earlier design asked an LLM to reproduce verbatim substrings, then treated a formatting discrepancy in one excerpt as a fatal error for all excerpts. Its fallback state did not contain a machine-readable reason. The observed failure was **not** a provider 401 or a structured-output parse failure.

## Fix on PR #19

1. Split reports deterministically into **bounded, source-offset-indexed** text spans (e.g., `M001`, `F013`).
2. The LLM emits only `ref` + `kind`; the program resolves each reference to the **exact original text**. Model paraphrasing or moving Markdown cannot change the saved evidence.
3. Ignore a small number of invalid/duplicate IDs, but still require coverage of every existing analyst report, a summary length budget, and **net size reduction**. If safety checks fail, use the complete original reports.
4. Record `research_evidence_error` in AgentState and saved logs when falling back; avoid a misleading promise of free-text retry. Limit candidate prompt characters so pathological runs fail before incurring a large summary call.
5. Add a minimal synthetic regression test containing the affected cash-flow claim and original Markdown, without committing the full private Trace.

## Limits and follow-up

- Original reports remain authoritative and preserved. Exact provenance **does not prove full decision-critical coverage** or independent truth; the summary is a selected view.
- The historical trace belongs to the **old** quote-copying schema. It proves the root cause; it cannot be replayed directly as an output of the new ID-based schema.
- Run a **summary-only** evaluation against frozen analyst reports before a full Trader rerun. Compare evidence coverage, stale Q3 data / missing figures, positive and negative facts, and the **net tokens** including the summary generation.
- Follow-up on the separate News Analyst tool-loop growth (3 to 8 LLM calls), which occurred upstream of this evidence node and should not be attributed to its fallback.
