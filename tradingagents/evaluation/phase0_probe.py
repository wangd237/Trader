"""Phase 0 observation probe: inspect LangGraph events without changing the workflow.

Only event metadata is retained. No prompts, tool arguments, responses, keys,
or full State values are persisted. This is not a production evaluator.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def _id(value: Any) -> str | None:
    return str(value) if value is not None else None


def _scope(checkpoint_ns: Any) -> str | None:
    """Compare nested callback namespaces to graph subgraph namespaces."""
    if not isinstance(checkpoint_ns, str) or not checkpoint_ns:
        return None
    return checkpoint_ns.split("|", 1)[0]


def _pairing(starts: list[Any], ends: list[Any]) -> dict[str, int]:
    """Count missing/duplicate spans in BOTH directions, not just dangling starts."""
    started, finished = Counter(starts), Counter(ends)
    return {
        "missing_end": sum((started - finished).values()),
        "missing_start": sum((finished - started).values()),
        "duplicate_start": sum(max(0, n - 1) for n in started.values()),
        "duplicate_end": sum(max(0, n - 1) for n in finished.values()),
        "missing_id": sum(
            value is None or (isinstance(value, tuple) and value[-1] is None)
            for value in [*starts, *ends]
        ),
    }


def _tool_key(scope: str | None, tool_call: dict[str, Any]) -> tuple[Any, ...]:
    return scope, _id(tool_call.get("tool_call_id")), tool_call.get("name")


def _tool_result_refs(value: Any) -> list[dict[str, Any]]:
    """Extract only ToolMessage correlation IDs; never retain response bodies."""
    if isinstance(value, dict):
        messages = value.get("messages") or []
    elif isinstance(value, (tuple, list)):
        messages = value
    else:
        messages = []
    if not isinstance(messages, (tuple, list)):
        messages = [messages]
    return [
        {"name": getattr(msg, "name", None), "tool_call_id": _id(msg.tool_call_id)}
        for msg in messages
        if getattr(msg, "tool_call_id", None) is not None
    ]


def _context(kwargs: dict[str, Any]) -> dict[str, Any]:
    metadata = kwargs.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "run_id": _id(kwargs.get("run_id")),
        "parent_run_id": _id(kwargs.get("parent_run_id")),
        "graph_node": metadata.get("langgraph_node"),
        "checkpoint_ns": metadata.get("langgraph_checkpoint_ns"),
    }


class Phase0TraceProbe(BaseCallbackHandler):
    """Thread-safe event observer for a single offline Graph execution."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def _emit(self, kind: str, **fields: Any) -> None:
        with self._lock:
            self.events.append(
                {
                    "event_id": f"event_{len(self.events) + 1:06d}",
                    "timestamp_ns": time.time_ns(),
                    "kind": kind,
                    **fields,
                }
            )

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self._emit("llm_start", **_context(kwargs))

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._emit("llm_start", **_context(kwargs))

    def _llm_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        context = _context(kwargs)
        if not context["checkpoint_ns"]:
            # on_llm_end does not necessarily repeat start-side metadata.
            # Rejoin via callback run_id, not node name or event timestamp.
            with self._lock:
                for event in reversed(self.events):
                    if event["kind"] == "llm_start" and event["run_id"] == context["run_id"]:
                        context["checkpoint_ns"] = event.get("checkpoint_ns")
                        context["graph_node"] = event.get("graph_node")
                        break
        return context

    def on_llm_end(self, response, **kwargs) -> None:
        tool_requests = []
        for generations in getattr(response, "generations", ()):
            for generation in generations:
                message = getattr(generation, "message", None)
                for call in getattr(message, "tool_calls", ()) or ():
                    tool_requests.append(
                        {"name": call.get("name"), "tool_call_id": call.get("id")}
                    )
        self._emit("llm_end", tool_requests=tool_requests, **self._llm_context(kwargs))

    def on_llm_error(self, error, **kwargs) -> None:
        self._emit("llm_error", error_type=type(error).__name__, **self._llm_context(kwargs))

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        # Never persist input_str: it may contain market data or credentials.
        name = serialized.get("name") if isinstance(serialized, dict) else None
        self._emit(
            "tool_start",
            tool_name=name or kwargs.get("name"),
            tool_call_id=_id(kwargs.get("tool_call_id")),
            **_context(kwargs),
        )

    def on_tool_end(self, output, **kwargs) -> None:
        # A completed callback is not proof that the data was valid.
        self._emit(
            "tool_end",
            tool_call_id=_id(getattr(output, "tool_call_id", None)),
            **_context(kwargs),
        )

    def on_tool_error(self, error, **kwargs) -> None:
        self._emit("tool_error", error_type=type(error).__name__, **_context(kwargs))

    def record_graph_event(self, namespace: Any, mode: str, payload: Any) -> None:
        """Consume native graph.stream(..., subgraphs=True) events."""
        ns = list(namespace) if isinstance(namespace, (tuple, list)) else []
        if mode == "tasks":
            if not isinstance(payload, dict):
                self._emit("unrecognized_graph_event", namespace=ns, stream_mode=mode)
                return
            if "result" in payload or ("error" in payload and "input" not in payload):
                kind = "graph_task_end"
            elif "input" in payload:
                kind = "graph_task_start"
            else:
                kind = "unrecognized_graph_event"
            self._emit(
                kind,
                namespace=ns,
                task_id=_id(payload.get("id")),
                node=payload.get("name"),
                error_type=(
                    type(payload["error"]).__name__
                    if payload.get("error") is not None
                    else None
                ),
                task_result_type=type(payload.get("result")).__name__ if kind == "graph_task_end" else None,
                task_result_fields=(
                    sorted(payload["result"])
                    if kind == "graph_task_end" and isinstance(payload.get("result"), dict)
                    else []
                ),
                tool_results=(
                    _tool_result_refs(payload.get("result"))
                    if kind == "graph_task_end" and payload.get("name") == "tools"
                    else []
                ),
            )
        elif mode == "updates":
            if not isinstance(payload, dict):
                self._emit("unrecognized_graph_event", namespace=ns, stream_mode=mode)
                return
            for node, update in payload.items():
                # Some LangGraph versions emit a None update after a tool
                # task. Keep both event channels, without assuming coverage.
                results = _tool_result_refs(update) if str(node) == "tools" else []
                self._emit(
                    "state_update",
                    namespace=ns,
                    node=str(node),
                    state_fields=sorted(update) if isinstance(update, dict) else [],
                    update_type=type(update).__name__,
                    message_types=(
                        [type(m).__name__ for m in update.get("messages", ())]
                        if isinstance(update, dict) and isinstance(update.get("messages"), list)
                        else []
                    ),
                    tool_results=results,
                )

    def coverage(self) -> dict[str, Any]:
        """Summarize full-graph, bidirectional lifecycle and ToolCall evidence."""
        counts = Counter(e["kind"] for e in self.events)
        cb = [e for e in self.events if e["kind"].startswith(("llm_", "tool_"))]

        graph_events = [e for e in self.events if e["kind"].startswith("graph_task_")]
        graph_starts = [
            (tuple(e["namespace"]), e["task_id"])
            for e in graph_events if e["kind"] == "graph_task_start"
        ]
        graph_ends = [
            (tuple(e["namespace"]), e["task_id"])
            for e in graph_events if e["kind"] == "graph_task_end"
        ]

        def callback_pairing(prefix: str) -> dict[str, int]:
            return _pairing(
                [e["run_id"] for e in self.events if e["kind"] == prefix + "_start"],
                [
                    e["run_id"] for e in self.events
                    if e["kind"] in (prefix + "_end", prefix + "_error")
                ],
            )

        # Distinguish data about an actual ToolNode result from the model's
        # intent. IDs may repeat across analyst subgraphs; the scope matters.
        requests = Counter(
            _tool_key(_scope(e.get("checkpoint_ns")), call)
            for e in self.events if e["kind"] == "llm_end"
            for call in e.get("tool_requests", ())
        )
        results = Counter(
            _tool_key(e["namespace"][0] if e.get("namespace") else None, result)
            for e in self.events if e["kind"] == "graph_task_end"
            for result in e.get("tool_results", ())
        )
        # Prefer task completions; updates can duplicate the same evidence.
        if not results:
            results = Counter(
                _tool_key(e["namespace"][0] if e.get("namespace") else None, result)
                for e in self.events if e["kind"] == "state_update"
                for result in e.get("tool_results", ())
            )
        matched = requests & results
        return {
            "counts": dict(sorted(counts.items())),
            "top_level_completed_nodes": sorted({
                e["node"] for e in graph_events
                if e["kind"] == "graph_task_end"
                and not e["namespace"] and e.get("node")
            }),
            "subgraph_namespaces": sorted({
                "/".join(e["namespace"]) for e in self.events if e.get("namespace")
            }),
            "callback_parent_links": sum(e.get("parent_run_id") is not None for e in cb),
            "callback_graph_node_metadata": sum(e.get("graph_node") is not None for e in cb),
            "graph_pairing": _pairing(graph_starts, graph_ends),
            "llm_pairing": callback_pairing("llm"),
            "tool_pairing": callback_pairing("tool"),
            "tool_task_diagnostics": [
                {
                    "namespace": "/".join(e.get("namespace") or []),
                    "task_result_type": e.get("task_result_type"),
                    "task_result_fields": e.get("task_result_fields"),
                    "tool_result_refs": e.get("tool_results"),
                }
                for e in self.events
                if e["kind"] == "graph_task_end" and e.get("node") == "tools"
            ],
            "tool_node_diagnostics": [
                {
                    "namespace": "/".join(e.get("namespace") or []),
                    "update_type": e.get("update_type"),
                    "state_fields": e.get("state_fields"),
                    "message_types": e.get("message_types"),
                }
                for e in self.events
                if e["kind"] == "state_update" and e.get("node") == "tools"
            ],
            "llm_scope_diagnostics": [
                {
                    "kind": e["kind"],
                    "scope": _scope(e.get("checkpoint_ns")),
                    "graph_node": e.get("graph_node"),
                }
                for e in self.events if e["kind"] in ("llm_start", "llm_end")
            ],
            "tool_call_linkage": {
                "requested": requests.total(),
                "result_messages": results.total(),
                "matched": matched.total(),
                "unmatched_requests": (requests - results).total(),
                "unmatched_results": (results - requests).total(),
                "unscoped_requests": sum(n for (scope, _, _), n in requests.items() if not scope),
                "unscoped_results": sum(n for (scope, _, _), n in results.items() if not scope),
            },
            "limitations": [
                "Graph task IDs and callback run IDs are different identifiers.",
                "InjectedState effective arguments are not exposed by this probe.",
                "Direct provider access inside Sentiment Analyst is not a ToolNode callback.",
                "Completed tool callbacks do not establish data validity.",
                "Prompts, tool inputs, responses and State values are deliberately omitted.",
            ],
        }

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            for event in self.events:
                out.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
