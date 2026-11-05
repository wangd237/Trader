"""Conservative runtime detection of repeated tool rounds with no new results.

This reads only the current analyst's private message history. It does not store
tool inputs or outputs, and never classifies a repeated request as wasteful
unless the returned content is also identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _request_key(requests: list[dict]) -> tuple:
    """Compare tool requests without their per-turn tool-call IDs."""
    return tuple((call.get("name"), _digest(call.get("args"))) for call in requests)


def repeats_warned_tool_request(messages: Sequence[BaseMessage], warning_prefix: str) -> bool:
    """Detect an unchanged batch immediately after a one-time corrective warning.

    A tool batch with different arguments or names is a genuine correction and
    is allowed. Do not apply an old warning to a later, unrelated batch.
    """
    if not messages or not isinstance(messages[-1], AIMessage):
        return False
    latest = messages[-1]
    if not latest.tool_calls:
        return False
    for index in range(len(messages) - 2, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage) and message.tool_calls:
            # Another tool request has already happened since the warning.
            return False
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            if message.content.startswith(warning_prefix):
                previous = next(
                    (
                        m for m in reversed(messages[:index])
                        if isinstance(m, AIMessage) and m.tool_calls
                    ),
                    None,
                )
                return previous is not None and _request_key(previous.tool_calls) == _request_key(
                    latest.tool_calls
                )
    return False


def consecutive_identical_tool_rounds(messages: Sequence[BaseMessage]) -> int:
    """Count matching completed tool rounds at the end of analyst history.

    A round is one AI message requesting one or more tools followed immediately
    by all corresponding ToolMessages. Both the calls (name + arguments) and the
    returned content/status must match. Tool call IDs may differ between rounds.
    Incomplete or malformed rounds break the streak rather than being guessed.
    """
    signatures: list[tuple | None] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        calls = message.tool_calls
        ids = [call.get("id") for call in calls]
        if (
            any(not isinstance(call_id, str) or not call_id for call_id in ids)
            or len(ids) != len(set(ids))
            or any(not call.get("name") for call in calls)
        ):
            signatures.append(None)
            continue

        replies = messages[index + 1 : index + 1 + len(calls)]
        if len(replies) != len(calls) or not all(
            isinstance(reply, ToolMessage) for reply in replies
        ):
            signatures.append(None)
            continue
        results = {reply.tool_call_id: reply for reply in replies}
        if len(results) != len(calls) or set(results) != set(ids):
            signatures.append(None)
            continue

        signatures.append(
            tuple(
                (
                    call["name"],
                    _digest(call.get("args")),
                    _digest({
                        "content": results[call["id"]].content,
                        "status": getattr(results[call["id"]], "status", None),
                    }),
                )
                for call in calls
            )
        )

    if not signatures or signatures[-1] is None:
        return 0
    last = signatures[-1]
    streak = 0
    for signature in reversed(signatures):
        if signature != last:
            break
        streak += 1
    return streak



# Opt-in by tool type. News, live market probabilities and today's prices must
# always reach their provider rather than a previous response.
HISTORICAL_REUSABLE_TOOLS = frozenset({"get_stock_data", "get_indicators"})


def cached_historical_tool_messages(
    messages: Sequence[BaseMessage], *, historical_run: bool,
) -> list[ToolMessage] | None:
    """Reuse an identical completed *historical* tool batch within one analyst run.

    The last message must contain the pending calls. Only whole batches made of
    explicitly stable, read-only historical tools qualify; partial batch reuse
    and failed/empty results intentionally fall back to the real ToolNode.
    Every replayed result receives the pending call's ID.
    """
    if not historical_run or not messages:
        return None
    pending = messages[-1]
    if not isinstance(pending, AIMessage) or not pending.tool_calls:
        return None

    calls = pending.tool_calls
    if (
        any(call.get("name") not in HISTORICAL_REUSABLE_TOOLS for call in calls)
        or any(not isinstance(call.get("id"), str) or not call["id"] for call in calls)
    ):
        return None

    wanted = _request_key(calls)
    for index in range(len(messages) - 2, -1, -1):
        previous = messages[index]
        if not isinstance(previous, AIMessage) or not previous.tool_calls:
            continue
        if _request_key(previous.tool_calls) != wanted:
            continue
        ids = [call.get("id") for call in previous.tool_calls]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            return None
        replies = messages[index + 1 : index + 1 + len(ids)]
        if len(replies) != len(ids) or not all(isinstance(r, ToolMessage) for r in replies):
            return None
        by_id = {reply.tool_call_id: reply for reply in replies}
        if len(by_id) != len(ids) or set(by_id) != set(ids):
            return None
        if any(
            getattr(by_id[i], "status", "success") != "success"
            or not isinstance(by_id[i].content, str)
            or not by_id[i].content.strip()
            or by_id[i].content.lstrip().upper().startswith(
                ("DATA_UNAVAILABLE:", "NO_DATA_AVAILABLE:", "ERROR:")
            )
            for i in ids
        ):
            return None
        return [
            ToolMessage(
                content=by_id[old_id].content,
                name=by_id[old_id].name or call["name"],
                tool_call_id=call["id"],
            )
            for old_id, call in zip(ids, calls, strict=True)
        ]
    return None
