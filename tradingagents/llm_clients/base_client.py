import warnings
from abc import ABC, abstractmethod
from typing import Any


def normalize_content(response):
    """将 LLM 返回内容规范化为普通字符串。

    OpenAI Responses API、Google Gemini 3 等可能返回类型化内容块列表，
    例如 reasoning 块与 text 块。下游 Agent 期望 response.content 是字符串，
    因此只提取和拼接文本块，丢弃推理及元数据块。"""
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


class BaseLLMClient(ABC):
    """LLM 客户端的抽象基类。"""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_provider_name(self) -> str:
        """返回用于告警消息的提供商名称。"""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """当模型不在该提供商的已知列表中时发出警告。"""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """返回已配置的 LLM 实例。"""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """校验当前客户端是否支持指定模型。"""
        pass
