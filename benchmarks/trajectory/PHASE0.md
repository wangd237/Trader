# Phase 0 — Trajectory coverage experiment

This is a read-only observation experiment, not a production evaluator.

## Run

From the repository root, run:

```bash
pytest -q -s tests/test_phase0_trace_probe.py
```

The test reuses the offline scripted model and mocked data fixture from `tests/test_graph_end_to_end.py`. It prints `PHASE0_COVERAGE`, without logging prompts, tool inputs, result bodies or full state values.

## Sources

- Native Graph `stream_mode=["tasks", "updates"]` with `subgraphs=True`: graph task boundaries, subgraph namespace and updated state key names.
- `BaseCallbackHandler`: LLM and ToolNode callback spans, IDs, metadata and model-requested tool names.

## Known noncoverage

- Graph task IDs and callback run IDs are not interchangeable.
- Direct data access inside Sentiment Analyst is not a model ToolNode call.
- The callback does not capture actual arguments after `InjectedState`.
- Successful tool callbacks may contain invalid/missing vendor data.
- State values, prompts, model responses, effective arguments and handoff evidence are intentionally excluded from Phase 0.
- A structured-output model backed by a RunnableLambda may not produce normal LLM callbacks; this probe tests the free-text full-Graph path.

Phase 1 should address missing causal links and evidence requirements only where the experiment demonstrates they are needed.
