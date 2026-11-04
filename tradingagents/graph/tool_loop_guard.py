"""Conservative runtime detection of repeated tool rounds with no new results.

This reads only the current analyst's private message history. It does not store
tool inputs or outputs, and never classifies a repeated request as wasteful
unless the returned content is also identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
