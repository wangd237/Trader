import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_GEMINI_VERSION = re.compile(r"^gemini-(\d+)\.(\d+)")


def _accepts_minimal_thinking(model: str) -> bool:
    """判断模型是否接受 ``thinking_level="minimal"``：仅 3.8 之前
    明确编号的 Flash 模型可用。Pro、3.8 及以上版本、
    以及会动态指向不同代际的无版本别名均视为不支持。"""
    model_lc = model.lower()
    match = _GEMINI_VERSION.match(model_lc)
    return bool(match) and "pro" not in model_lc and (
        (int(match.group(1)), int(match.group(2))) < (3, 8)
    )


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """具有标准化文本输出的 ChatGoogleGenerativeAI。

    Gemini 3 可能返回类型化内容块列表；这里统一转换成字符串，
    以便下游保持一致。"""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


# 与 OpenAI、Anthropic SDK 默认值一致的读取超时时间。
REQUEST_TIMEOUT_SECONDS = 600


class GoogleClient(BaseLLMClient):
    """Google Gemini 模型客户端。"""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """返回已配置的 ChatGoogleGenerativeAI 实例。"""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "temperature", "max_output_tokens",
                    "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # Gemini 调用如果未设置超时，连接卡住时可能一直等待；
        # 这里采用与 OpenAI、Anthropic SDK 相同的 600 秒上限（#1417）。
        llm_kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)

        # 将统一 api_key 参数映射成 Google 专属的 google_api_key。
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Gemini 3.x 使用字符串 ``thinking_level``，旧版 2.5 才使用整数
        # ``thinking_budget``。Pro、Gemini 3.8+ 和 -latest 别名会拒绝
        # "minimal" 并返回 400，因此使用更通用的 "low" 作为回退。
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if thinking_level == "minimal" and not _accepts_minimal_thinking(self.model):
                thinking_level = "low"
            llm_kwargs["thinking_level"] = thinking_level

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """验证 Google 模型名称。"""
        return validate_model("google", self.model)
