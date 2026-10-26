"""调用结构化输出 Agent 的共用辅助函数，并支持失败后降级。

Portfolio Manager、Trader、Research Manager 使用统一流程：
1. 创建 Agent 时调用 ``with_structured_output(Schema)`` 包装 LLM，
   让其输出 Pydantic 结构化实例。若提供商不支持结构化输出
   （少数旧版 Ollama 模型），则直接使用普通文本生成。
2. 运行时先执行结构化调用并渲染成 Markdown；若因 JSON 格式错误、
   提供商临时故障等原因失败，则回退到普通的 ``llm.invoke``，
   避免整体流程阻塞。

将该模式集中实现，既精简 Agent 工厂，也统一三类 Agent
触发回退时的警告日志。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 仅绑定 Schema 的结构化输出模式实际上只允许调用一个工具，
# 即 Schema 本身。若模型尝试调用搜索工具，会产生未知工具调用，
# 整个结构化结果就会被丢弃并进入文本回退。
# 因此此路径中的 Agent 会明确告诉模型只能使用规定 Schema，
# 而不能仅依赖工具绑定约束（#1130）。
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """返回 ``llm.with_structured_output(schema)``；不支持时返回 None。

    如果绑定失败，则记录警告，说明后续将持续使用普通文本输出，
    而不只是单次请求的临时回退。"""
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured(structured_llm: Any | None, prompt: Any, agent_name: str) -> T | None:
    """执行结构化模型调用；未配置或执行失败时返回 None。

    ``prompt`` 可以是底层 LLM 接受的任意格式，
    例如聊天请求的字符串或消息字典列表，
    调用方可将同一个值直接传入文本回退调用。"""
    if structured_llm is None:
        return None
    try:
        result = structured_llm.invoke(prompt)
        if result is None:
            # 思考模型可能直接返回文本而不调用预期的结构化工具，
            # 因此必须为这类情况提供回退处理。
            raise ValueError("structured output returned no parsed result")
        return result
    except Exception as exc:
        logger.warning(
            "%s: structured-output invocation failed (%s); retrying once as free text",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """执行结构化调用并渲染为 Markdown；任意失败均回退到普通文本生成。"""
    result = invoke_structured(structured_llm, prompt, agent_name)
    if result is not None:
        return render(result)
    return plain_llm.invoke(prompt).content
