# Phase 1 — Minimal Trajectory Collector + Rule Evaluator

Phase 1 extends the Phase 0 read-only observer without changing Trader's business nodes.
It does not add an LLM Judge, complex infrastructure, or investment-return scoring.

## Artifacts and evidence

One diagnostic run writes results/trajectory_eval/<run_id>/trace.jsonl and
evaluation.json. The run ID is distinct from callback_run_id and
parent_callback_run_id. It preserves case_id, attempt_id, analysis date, analyst
selection, model ToolCall name/ID, callbacks, Graph tasks and State field names.

Model-visible tool arguments are validated in memory against that analyst's
allowed tools, public input schema, and analysis-date cutoff. A per-run HMAC
fingerprint allows conservative duplicate candidates without persisting inputs.
No prompts, raw arguments, tool results, messages or complete State values are
persisted. ToolNode results are linked by scope + ToolCall ID + name.

## Reproduce

    pytest -q -s tests/test_trajectory_evaluation.py

The offline test uses the existing scripted model and mocked vendors. It prints
PHASE1_SUMMARY before its assertions. The Graph is actually executed.

Python API:

    from pathlib import Path
    from tradingagents.evaluation.collector import collect_trajectory
    report = collect_trajectory(
        graph, "NVDA", "2026-01-09",
        case_id="scenario_01", attempt_id="1",
        output_dir=Path("results/trajectory_eval"),
    )

The supplied graph must be a TradingAgentsGraph with an isolated cache and
Memory Log, checkpoint_enabled=False. The diagnostic runner does not call
record_decision(), but the normal Memory Log Graph node still executes and can
settle pending history: isolate data directories before live runs.

Re-evaluate saved traces without invoking any model or tools:

    python scripts/eval_trajectory.py results/trajectory_eval
    python scripts/eval_trajectory.py --strict results/trajectory_eval/<run_id>/trace.jsonl

## Deterministic rules

- task_completion: Graph finished, selected reports and decision fields are
  nonempty, rating is recognized. This is not an evidence-grounding evaluation.
- lifecycle_pairing: all Graph tasks, model and tool callbacks pair in both
  directions; orphan events and duplicates are flagged.
- tool_result_linkage: model requests and actual ToolMessage IDs match within
  the correct analyst subgraph.
- model_tool_intent: tool allowed, model-visible schema valid, and requested
  dates within the analysis window. These are model requests, not effective
  runtime arguments after InjectedState.
- tool_round_budget: rounds that request tools, not individual parallel calls;
  force-wrap-up event checked on reaching the budget.
- graph_route: all selected analysts and Memory Log precede debate and
  subsequent research/Trader/risk dependencies.

Each rule reports pass, fail or not_applicable plus evidence event IDs. No
weighted composite score is reported.

Possible repeated calls with matching analyst/tool/fingerprint are marked only
as candidate_requires_semantic_review, not as proven wasted work.

## Non-goals

Real-model benchmarks, Handoff semantic quality, factual grounding, market-data
validity, checkpoint recovery scoring, vendor error-text classification,
effective tool arguments and token-cost analysis are deliberately deferred.
