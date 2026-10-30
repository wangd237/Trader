"""Phase 1: opt-in trajectory capture using the Phase 0 Graph observer.

No prompts, raw arguments, tool responses or full State values are persisted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import tradingagents
from tradingagents.dataflows.config import run_config_context
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

from .phase0_probe import Phase0TraceProbe, _id, _scope
from .rules import evaluate_events

_DATE_KEYS = ("start_date", "end_date", "curr_date")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RATINGS = {"Buy", "Overweight", "Hold", "Underweight", "Sell"}


def _date_errors(args: dict[str, Any], trade_date: str) -> list[str]:
    errors = []
    dates: dict[str, date] = {}
    cutoff = date.fromisoformat(trade_date)
    for key in _DATE_KEYS:
        if key not in args or args[key] is None:
            continue
        value = args[key]
        if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
            errors.append(f"{key}:invalid_date")
            continue
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            errors.append(f"{key}:invalid_date")
            continue
        dates[key] = parsed
        if parsed > cutoff:
            errors.append(f"{key}:after_trade_date")
    if "start_date" in dates and "end_date" in dates:
        if dates["start_date"] > dates["end_date"]:
            errors.append("window:reversed")
    return errors


class TrajectoryCollector(Phase0TraceProbe):
    """Capture event metadata and validated model intent for one evaluation run."""

    def __init__(
        self,
        *,
        case_id: str,
        attempt_id: str,
        trade_date: str,
        selected_analysts: tuple[str, ...],
        max_tool_rounds: int,
    ) -> None:
        super().__init__()
        self.evaluation_run_id = uuid4().hex
        self.trade_date = trade_date
        self._fingerprint_key = secrets.token_bytes(32)
        self._available = {
            spec.agent_node: {tool.name: tool for tool in spec.tools}
            for key in selected_analysts
            for spec in (ANALYST_NODE_SPECS[key],)
        }
        self._emit(
            "evaluation_start",
            evaluation_run_id=self.evaluation_run_id,
            case_id=case_id,
            attempt_id=attempt_id,
            trade_date=trade_date,
            selected_analysts=list(selected_analysts),
            max_tool_rounds=max_tool_rounds,
            version=tradingagents.__version__,
        )

    def on_llm_end(self, response, **kwargs: Any) -> None:
        super().on_llm_end(response, **kwargs)
        context = self._llm_context(kwargs)
        scope = _scope(context.get("checkpoint_ns"))
        agent = scope.split(":", 1)[0] if scope else context.get("graph_node")
        allowed = self._available.get(agent)
        for generations in getattr(response, "generations", ()):
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                calls = list(getattr(message, "tool_calls", ()) or ())
                invalid = list(getattr(message, "invalid_tool_calls", ()) or ())
                for call in calls + invalid:
                    args = call.get("args")
                    valid_args = isinstance(args, dict) and call in calls
                    canonical = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
                    fingerprint = hmac.new(
                        self._fingerprint_key, canonical.encode("utf-8"), hashlib.sha256
                    ).hexdigest()[:20]
                    name = call.get("name")
                    authorized = allowed is not None and name in allowed
                    schema_valid = False
                    if valid_args and authorized:
                        try:
                            allowed[name].tool_call_schema.model_validate(args)
                            schema_valid = True
                        except (TypeError, ValueError):
                            pass
                    self._emit(
                        "tool_intent",
                        model_run_id=_id(kwargs.get("run_id")),
                        scope=scope,
                        agent=agent,
                        tool_name=name,
                        tool_call_id=_id(call.get("id")),
                        arg_fingerprint=fingerprint,
                        authorized=authorized,
                        schema_valid=schema_valid,
                        date_errors=_date_errors(args, self.trade_date) if valid_args else [],
                        malformed=not valid_args,
                    )

    def finish(self, final_state: dict[str, Any] | None, error: BaseException | None) -> None:
        state = final_state or {}
        meta = next(e for e in self.events if e["kind"] == "evaluation_start")
        reports = {
            ANALYST_NODE_SPECS[key].report_key: bool(
                str(state.get(ANALYST_NODE_SPECS[key].report_key) or "").strip()
            )
            for key in meta["selected_analysts"]
        }
        reports.update({
            key: bool(str(state.get(key) or "").strip())
            for key in ("investment_plan", "trader_investment_plan", "final_trade_decision")
        })
        self._emit(
            "evaluation_end",
            finished=error is None,
            error_type=type(error).__name__ if error else None,
            report_presence=reports,
            rating_valid=state.get("final_rating") in _RATINGS,
        )

    def save(self, directory: Path) -> dict[str, Any]:
        """Persist an independent JSONL trace and deterministic evaluation JSON."""
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "trace.jsonl").open("w", encoding="utf-8") as out:
            for event in self.events:
                row = dict(event)
                callback_id = row.pop("run_id", None)
                parent_id = row.pop("parent_run_id", None)
                row["run_id"] = self.evaluation_run_id
                if callback_id is not None:
                    row["callback_run_id"] = callback_id
                if parent_id is not None:
                    row["parent_callback_run_id"] = parent_id
                out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        report = evaluate_events(self.events)
        (directory / "evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Load safe persisted evidence into the Phase 0 callback naming convention."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event.pop("run_id", None)
        if "callback_run_id" in event:
            event["run_id"] = event.pop("callback_run_id")
        if "parent_callback_run_id" in event:
            event["parent_run_id"] = event.pop("parent_callback_run_id")
        events.append(event)
    return events


def collect_trajectory(
    graph: Any,
    ticker: str,
    trade_date: str,
    *,
    case_id: str,
    attempt_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Observe the existing Graph without writing a decision or managing Checkpoint.

    Use an isolated cache and Memory Log. Failures are captured as evidence;
    they are not silently treated as successful runs.
    """
    if graph.config.get("checkpoint_enabled"):
        raise ValueError("Phase 1 diagnostic runs require checkpoint_enabled=False")
    collector = TrajectoryCollector(
        case_id=case_id,
        attempt_id=attempt_id,
        trade_date=trade_date,
        selected_analysts=tuple(graph.selected_analysts),
        max_tool_rounds=graph.config["max_tool_rounds"],
    )
    config = graph.propagator.get_graph_args(callbacks=[collector])["config"]
    context = run_config_context(graph.config)
    initial = graph.create_run_state(ticker, trade_date)
    final_state = None
    error = None
    stream = context.run(
        graph.graph.stream,
        initial,
        config=config,
        stream_mode=["tasks", "updates", "values"],
        subgraphs=True,
    )
    try:
        while (step := context.run(next, stream, None)) is not None:
            namespace, mode, payload = step
            if not namespace and mode == "values" and isinstance(payload, dict):
                final_state = payload
            else:
                collector.record_graph_event(namespace, mode, payload)
    except Exception as exc:
        error = exc
    finally:
        context.run(stream.close)
    collector.finish(final_state, error)
    return collector.save(Path(output_dir) / collector.evaluation_run_id)
