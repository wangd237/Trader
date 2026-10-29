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

    def on_llm_end(self, response, **kwargs) -> None:
        tool_requests = []
        for generations in getattr(response, "generations", ()):
            for generation in generations:
                message = getattr(generation, "message", None)
                for call in getattr(message, "tool_calls", ()) or ():
                    tool_requests.append(
                        {"name": call.get("name"), "tool_call_id": call.get("id")}
                    )
        self._emit("llm_end", tool_requests=tool_requests, **_context(kwargs))

    def on_llm_error(self, error, **kwargs) -> None:
        self._emit("llm_error", error_type=type(error).__name__, **_context(kwargs))

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
            )
        elif mode == "updates":
            if not isinstance(payload, dict):
                self._emit("unrecognized_graph_event", namespace=ns, stream_mode=mode)
                return
            for node, update in payload.items():
                self._emit(
                    "state_update",
                    namespace=ns,
                    node=str(node),
                    state_fields=sorted(update) if isinstance(update, dict) else [],
                )

    def coverage(self) -> dict[str, Any]:
        """Report observed evidence and known gaps, never infer missing events."""
        counts = Counter(e["kind"] for e in self.events)
        cb = [e for e in self.events if e["kind"].startswith(("llm_", "tool_"))]
        tool_starts = {e["run_id"] for e in self.events if e["kind"] == "tool_start"}
        tool_ends = {
            e["run_id"] for e in self.events if e["kind"] in ("tool_end", "tool_error")
        }
        llm_starts = {e["run_id"] for e in self.events if e["kind"] == "llm_start"}
        llm_ends = {
            e["run_id"] for e in self.events if e["kind"] in ("llm_end", "llm_error")
        }
        return {
            "counts": dict(sorted(counts.items())),
            "subgraph_namespaces": sorted(
                {"/".join(e["namespace"]) for e in self.events if e.get("namespace")}
            ),
            "callback_parent_links": sum(e.get("parent_run_id") is not None for e in cb),
            "callback_graph_node_metadata": sum(e.get("graph_node") is not None for e in cb),
            "llm_unmatched_start": len(llm_starts - llm_ends),
            "tool_unmatched_start": len(tool_starts - tool_ends),
            "tool_end_with_tool_call_id": sum(
                bool(e.get("tool_call_id"))
                for e in self.events
                if e["kind"] == "tool_end"
            ),
            "limitations": [
                "Graph task IDs and callback run IDs are different identifiers.",
                "Callback metadata may not uniquely map an agent subgraph to an LLM span.",
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
