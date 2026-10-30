"""Phase 1 deterministic trajectory rules; no LLM Judge and no outcome-profit metric."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

from .phase0_probe import Phase0TraceProbe, _scope


def _check(status: str, details: list[str], event_ids: list[str]) -> dict[str, Any]:
    return {"status": status, "details": details, "event_ids": sorted(set(event_ids))}


def _node_order(events: list[dict[str, Any]], selected: list[str]) -> dict[str, Any]:
    completed = [
        e for e in events
        if e["kind"] == "graph_task_end" and not e.get("namespace") and e.get("node")
    ]
    positions: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(completed):
        positions[e["node"]].append(i)
    analysts = [ANALYST_NODE_SPECS[key].agent_node for key in selected]
    requirements = analysts + [
        "Memory Log", "Bull Researcher", "Bear Researcher", "Research Manager",
        "Trader", "Aggressive Analyst", "Conservative Analyst",
        "Neutral Analyst", "Portfolio Manager",
    ]
    missing = [node for node in requirements if node not in positions]
    violations = []
    if not missing:
        checkpoints = [
            (max(positions[n][-1] for n in analysts + ["Memory Log"]), positions["Bull Researcher"][0], "analyst_barrier"),
            (
                max(positions[n][-1] for n in ("Bull Researcher", "Bear Researcher")),
                positions["Research Manager"][0],
                "research_debate",
            ),
            (positions["Research Manager"][-1], positions["Trader"][0], "research_to_trader"),
            (positions["Trader"][-1], positions["Aggressive Analyst"][0], "trader_to_risk"),
            (
                max(positions[n][-1] for n in (
                    "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"
                )),
                positions["Portfolio Manager"][0],
                "risk_to_manager",
            ),
        ]
        violations = [label for before, after, label in checkpoints if before >= after]
    bad = bool(missing or violations)
    evidence = [e["event_id"] for e in completed]
    return _check(
        "fail" if bad else "pass",
        [*(f"missing_node:{name}" for name in missing), *(f"order:{x}" for x in violations)],
        evidence if bad else [],
    )


def _budget(events: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    by_scope: dict[str, set[str]] = defaultdict(set)
    for e in events:
        if e["kind"] != "llm_end" or not e.get("tool_requests"):
            continue
        scope = _scope(e.get("checkpoint_ns"))
        if scope is None or not e.get("run_id"):
            return _check("fail", ["unscoped_tool_round"], [e["event_id"]])
        by_scope[scope].add(e["run_id"])
    if not by_scope:
        return _check("not_applicable", ["no_model_tool_rounds"], [])
    wrapups = {
        (e["namespace"][0] if e.get("namespace") else None)
        for e in events if e["kind"] == "graph_task_end" and e.get("node") == "wrap_up"
    }
    errors = []
    for scope, run_ids in by_scope.items():
        if len(run_ids) > limit:
            errors.append(f"over_budget:{scope}")
        if len(run_ids) == limit and scope not in wrapups:
            errors.append(f"missing_forced_wrapup:{scope}")
    return _check("fail" if errors else "pass", errors, [])


def _candidate_duplicates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first: dict[tuple[str, str, str], str] = {}
    duplicates = []
    for e in events:
        if e["kind"] != "tool_intent":
            continue
        scope = e.get("scope")
        fingerprint = e.get("arg_fingerprint")
        name = e.get("tool_name")
        if not scope or not name or not fingerprint:
            continue
        key = (scope, name, fingerprint)
        if key in first:
            duplicates.append({
                "first_event_id": first[key],
                "duplicate_event_id": e["event_id"],
                "agent": e.get("agent"),
                "tool_name": name,
                "status": "candidate_requires_semantic_review",
            })
        else:
            first[key] = e["event_id"]
    return duplicates


def evaluate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate safe event metadata; unknown evidence must not become a pass."""
    starts = [e for e in events if e["kind"] == "evaluation_start"]
    ends = [e for e in events if e["kind"] == "evaluation_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("trace requires exactly one evaluation_start and evaluation_end")
    start, end = starts[0], ends[0]

    # Reuse the Phase 0 pairing and tool-message causal evidence.
    probe = Phase0TraceProbe()
    probe.events = events
    observed = probe.coverage()
    pairing_names = ("graph_pairing", "llm_pairing", "tool_pairing")
    pairing_errors = [
        f"{kind}:{field}={count}"
        for kind in pairing_names
        for field, count in observed[kind].items()
        if count
    ]
    task_complete = (
        bool(end.get("finished"))
        and bool(end.get("rating_valid"))
        and bool(end.get("report_presence"))
        and all(end["report_presence"].values())
    )
    missing_reports = [
        key for key, value in end.get("report_presence", {}).items() if not value
    ]
    task_details = []
    if not end.get("finished"):
        task_details.append(f"graph_error:{end.get('error_type') or 'unknown'}")
    if not end.get("rating_valid"):
        task_details.append("rating_invalid_or_missing")
    task_details.extend(f"missing_report:{key}" for key in missing_reports)
    if not end.get("report_presence"):
        task_details.append("missing_report_evidence")

    link = observed["tool_call_linkage"]
    link_errors = [
        f"{field}:{link[field]}"
        for field in ("unmatched_requests", "unmatched_results", "unscoped_requests", "unscoped_results")
        if link[field]
    ]
    intents = [e for e in events if e["kind"] == "tool_intent"]
    invalid = [
        e for e in intents
        if not e.get("authorized") or not e.get("schema_valid") or e.get("malformed")
    ]
    bad_dates = [e for e in intents if e.get("date_errors")]
    model_call_errors = [
        f"{e['event_id']}:{'unauthorized' if not e.get('authorized') else 'invalid_schema'}"
        for e in invalid
    ]
    model_call_errors.extend(
        f"{e['event_id']}:{','.join(e['date_errors'])}" for e in bad_dates
    )
    rules = {
        "task_completion": _check(
            "pass" if task_complete else "fail", task_details, [end["event_id"]]
        ),
        "lifecycle_pairing": _check(
            "fail" if pairing_errors else "pass", pairing_errors, []
        ),
        "tool_result_linkage": _check(
            "fail" if link_errors else ("pass" if link["requested"] else "not_applicable"),
            link_errors if link_errors else (["no_tool_requests"] if not link["requested"] else []),
            [e["event_id"] for e in events if e["kind"] == "llm_end" and e.get("tool_requests")]
            if link_errors else [],
        ),
        "model_tool_intent": _check(
            "fail" if model_call_errors else ("pass" if intents else "not_applicable"),
            model_call_errors if model_call_errors else (["no_model_tool_intents"] if not intents else []),
            sorted({e["event_id"] for e in invalid + bad_dates}),
        ),
        "tool_round_budget": _budget(events, int(start["max_tool_rounds"])),
        "graph_route": (
            _node_order(events, list(start["selected_analysts"]))
            if end.get("finished")
            else _check("not_applicable", ["graph_interrupted"], [])
        ),
    }
    candidates = _candidate_duplicates(events)
    statuses = Counter(v["status"] for v in rules.values())
    return {
        "run_id": start["evaluation_run_id"],
        "case_id": start["case_id"],
        "attempt_id": start["attempt_id"],
        "status": "fail" if statuses["fail"] else "pass",
        "metrics": {
            "llm_calls": observed["counts"].get("llm_start", 0),
            "tool_calls": observed["counts"].get("tool_start", 0),
            "candidate_duplicate_tool_calls": len(candidates),
        },
        "rules": rules,
        "candidate_duplicates": candidates,
        "coverage": {
            "counts": observed["counts"],
            "graph_pairing": observed["graph_pairing"],
            "llm_pairing": observed["llm_pairing"],
            "tool_pairing": observed["tool_pairing"],
            "tool_call_linkage": observed["tool_call_linkage"],
        },
    }
