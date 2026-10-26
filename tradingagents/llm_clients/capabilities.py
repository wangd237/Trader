"""面向 OpenAI 兼容提供商的声明式模型能力表。

统一描述不同模型会拒绝哪些 API 参数，以及需要哪种结构化输出方式。
客户端通过 ``get_capabilities(model_name)`` 查表，
避免到处写模型名称分支。新增模型或兼容性特例只需修改本表，
不需要改客户端逻辑。
该设计参考 DeepSeek 集成文档中按模型声明兼容标记的方式。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

StructuredMethod = Literal[
    "function_calling",  # uses tools; respects supports_tool_choice
    "json_mode",         # uses response_format={"type":"json_object"}
    "json_schema",       # uses response_format={"type":"json_schema",...}
    "none",              # no structured output available; caller falls back to free-text
]


@dataclass(frozen=True)
class ModelCapabilities:
    """描述 OpenAI 兼容模型在 API 层能够接受的参数与功能。"""

    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: StructuredMethod
    # DeepSeek 思考模式要求在后续请求中回传上一轮 assistant 的
    # reasoning_content，否则可能返回 HTTP 400。
    requires_reasoning_content_roundtrip: bool = False
    # MiniMax M2.x 推理模型需要 ``reasoning_split=True``，
    # 将 <think> 内容放入 ``reasoning_details``，避免污染 ``content``。
    # 非推理 MiniMax 模型不接受该参数，因此仅对确实使用它的模型启用（#826）。
    requires_reasoning_split: bool = False


# DeepSeek 思考模型接受 ``tools`` 列表，但拒绝 ``tool_choice`` 参数
# （官方集成说明及 #678 的 400 错误）。
# 官方工具调用示例只传 ``tools=[...]``；因此将
# supports_tool_choice 设置为 False，让客户端自动去掉该参数。
_DEEPSEEK_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_content_roundtrip=True,
)

_DEEPSEEK_CHAT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
)

# MiniMax M2.x 模型接受 tools 列表，但 tool_choice 仅可设置
# "none" 或 "auto"。LangChain 的 function_calling 会传函数描述字典，
# 导致 HTTP 400，问题与 DeepSeek 类似。
# 因此 supports_tool_choice=False 禁止发送该参数，
# 但仍将 Schema 作为工具发送。json_mode 的 response_format
# 只适用于 MiniMax-Text-01，不适用于 M2.x。
_MINIMAX_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=False,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_split=True,
)

_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
)


# 精确 ID 匹配优先于模式匹配。
_BY_ID: dict[str, ModelCapabilities] = {
    "deepseek-chat": _DEEPSEEK_CHAT,
    "deepseek-reasoner": _DEEPSEEK_THINKING,
    "deepseek-flash": _DEEPSEEK_THINKING,
    "deepseek-v4-flash": _DEEPSEEK_THINKING,
    "deepseek-v4-pro": _DEEPSEEK_THINKING,
    # MiniMax 官方模型列表，来源：platform.minimax.io/docs/api-reference/text-openai-api。
    "MiniMax-M2.7": _MINIMAX_THINKING,
    "MiniMax-M2.7-highspeed": _MINIMAX_THINKING,
    "MiniMax-M2.5": _MINIMAX_THINKING,
    "MiniMax-M2.5-highspeed": _MINIMAX_THINKING,
    "MiniMax-M2.1": _MINIMAX_THINKING,
    "MiniMax-M2.1-highspeed": _MINIMAX_THINKING,
    "MiniMax-M2": _MINIMAX_THINKING,
}

# 为将来版本设置兼容性匹配模式。新的 ``deepseek-v5-*``、
# ``deepseek-reasoner-*``、``deepseek-flash-*`` 和 ``MiniMax-M3*``
# 将自动继承对应的推理模型限制。``deepseek-flash`` 是模型列表中
# V4.1 Flash 的无版本别名，同样需要禁用 ``tool_choice``。
_BY_PATTERN: list[tuple[re.Pattern[str], ModelCapabilities]] = [
    (re.compile(r"^deepseek-v\d"), _DEEPSEEK_THINKING),
    (re.compile(r"^deepseek-reasoner"), _DEEPSEEK_THINKING),
    (re.compile(r"^deepseek-flash"), _DEEPSEEK_THINKING),
    (re.compile(r"^MiniMax-M\d"), _MINIMAX_THINKING),
]


def get_capabilities(model_name: str) -> ModelCapabilities:
    """按精确模型 ID、模式匹配、默认值的优先级查询能力。"""
    # OpenRouter 将官方 DeepSeek 模型写成 ``deepseek/<id>``；
    # 去掉此命名空间前缀后，可复用原生 DeepSeek 的兼容规则。
    # 例如 ``deepseek/deepseek-v4-flash`` 必须像原生模型一样禁用
    # ``tool_choice``，而非错误落到默认配置（#1199）。
    # 只对官方命名空间做此处理；其他发布者微调版本的行为未知，
    # 因此继续采用默认能力配置。
    if model_name.startswith("deepseek/"):
        model_name = model_name.removeprefix("deepseek/")

    if model_name in _BY_ID:
        return _BY_ID[model_name]
    for pattern, caps in _BY_PATTERN:
        if pattern.match(model_name):
            return caps
    return _DEFAULT
