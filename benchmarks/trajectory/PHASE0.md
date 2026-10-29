# Phase 0 — Trajectory coverage experiment

This is a read-only observation experiment, not a production evaluator.

## Run

From the repository root, run:

```bash
pytest -q -s tests/test_phase0_trace_probe.py
```

The test reuses the offline scripted model and mocked data fixture from `tests/test_graph_end_to_end.py`. It prints `PHASE0_COVERAGE` before acceptance assertions, without logging prompts, tool inputs, result bodies or full state values. The Python 3.11 CI job re-runs the probe with `-s`, exposing the actual coverage summary in Actions logs.

## Sources

- Native Graph `stream_mode=["tasks", "updates"]` with `subgraphs=True`: graph task boundaries, subgraph namespace and updated state key names.
- `BaseCallbackHandler`: LLM and ToolNode callback spans, IDs, metadata and model-requested tool names.

## Evidence-based acceptance

The full offline Graph test checks:

1. Every selected top-level analyst, Memory Log, debate and decision node completes.
2. Graph tasks and LLM/Tool callbacks pair bidirectionally using namespace/task ID or callback run ID respectively. Orphan starts, orphan ends, duplicates and missing IDs fail.
3. Model-requested (subgraph, tool_call_id, tool_name) matches actual ToolMessage metadata from the tools-node State update. IDs may repeat across parallel subgraphs, so matching by ID alone is insufficient.

## Known noncoverage

- Graph task IDs and callback run IDs are not interchangeable.
- Direct data access inside Sentiment Analyst is not a model ToolNode call.
- The callback does not capture actual arguments after `InjectedState`.
- Successful tool callbacks may contain invalid/missing vendor data.
- State values, prompts, model responses, effective arguments and handoff evidence are intentionally excluded from Phase 0.
- A structured-output model backed by a RunnableLambda may not produce normal LLM callbacks; this probe tests the free-text full-Graph path.

Phase 0 aims to prove causal links between model tool requests and ToolNode result messages within each analyst namespace. It does not prove the validity of external data, effective arguments, or the complete prompt/evidence Handoff chain. Phase 1 should address only empirically demonstrated gaps.
