"""执行分析师的一轮工具交互；工具轮数耗尽后执行收尾轮次。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

WRAP_UP = (
    "You have used every tool round this report allows. Write your final report "
    "now from the tool results above, and say which data you could not retrieve."
)


def take_turn(prompt, llm, tools, messages):
    """生成分析师的回复与报告；工具调用期间报告保持为空。

    收尾时不再向模型绑定工具，因为模型即使被要求结束，也可能继续调用。
    历史工具调用与结果会转为文本传入，避免部分提供商拒绝接受
    包含工具调用记录、却未绑定工具的对话历史。"""
    if messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == WRAP_UP:
        # Prompt 只能列出实际提供的工具；如果提到不可调用的工具，
        # 模型仍可能尝试调用，部分提供商会因此返回空响应。
        prompt = prompt.partial(tool_names="none; your tool rounds are spent")
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
