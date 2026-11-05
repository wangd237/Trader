"""执行分析师的一轮工具交互；工具轮数耗尽后执行收尾轮次。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

WRAP_UP = (
    "You have used every tool round this report allows. Write your final report "
    "now from the tool results above, and say which data you could not retrieve."
)

WRAP_UP_REPEATED = (
    "Tool access is now disabled because consecutive identical tool requests "
    "returned unchanged results."
)

TOOL_LOOP_WARNING = (
    "Repeated tool requests have returned unchanged results."
)
BLOCKED_REPEAT_RESULT = (
    "SKIPPED_REPEAT: This identical tool request was blocked after the corrective "
    "warning. No tool was executed."
)


def take_turn(prompt, llm, tools, messages):
    """生成分析师的回复与报告；工具调用期间报告保持为空。

    收尾时不再向模型绑定工具，因为模型即使被要求结束，也可能继续调用。
    历史工具调用与结果会转为文本传入，避免部分提供商拒绝接受
    包含工具调用记录、却未绑定工具的对话历史。"""
    latest = messages[-1] if messages else None
    exhausted = isinstance(latest, HumanMessage) and latest.content == WRAP_UP
    repeated = (
        isinstance(latest, HumanMessage)
        and isinstance(latest.content, str)
        and latest.content.startswith(WRAP_UP_REPEATED)
    )
    if exhausted or repeated:
        # Prompt 与实际工具接口都不再提供工具，不依赖模型服从文字指令。
        reason = "your tool rounds are spent" if exhausted else "repeated requests returned no new data"
        prompt = prompt.partial(tool_names=f"none; {reason}")
        result = (prompt | llm).invoke(_as_text(messages))
        return result, result.content
    result = (prompt | llm.bind_tools(tools)).invoke(messages)
    return result, "" if result.tool_calls else result.content


def _as_text(messages):
    """将历史工具调用及其返回结果整理为普通文本消息。"""
    written = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            said = message.content if isinstance(message.content, str) else ""
            calls = "; ".join(f"{c['name']}({json.dumps(c['args'], default=str)})" for c in message.tool_calls)
            written.append(AIMessage(content=f"{said}\n[Called {calls}]".strip()))
        elif isinstance(message, ToolMessage):
            written.append(HumanMessage(content=f"[{message.name or 'Tool'} returned]\n{message.content}"))
        else:
            written.append(message)
    return written
