import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient, normalize_content
from .capabilities import get_capabilities
from .validators import validate_model

logger = logging.getLogger(__name__)


class NormalizedChatOpenAI(ChatOpenAI):
    """支持标准化内容输出与模型能力感知绑定的 ChatOpenAI。

    Responses API 返回 reasoning、text 等不同类型的内容块，
    ``invoke`` 将其统一成字符串以便下游使用。
    ``with_structured_output`` 根据 ``capabilities.get_capabilities``
    选择结构化输出方式及是否允许发送 ``tool_choice``。
    对于 DeepSeek V4、reasoner 等拒绝 ``tool_choice`` 的模型，
    仍把 Schema 绑定为工具，但不发送该参数。
    推理内容回传等提供商特例放在子类中，保持基础类简洁。"""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )
        method = method or caps.preferred_structured_method
        # 模型拒绝 ``tool_choice`` 时，禁止 LangChain 自动传入该参数；
        # Schema 仍绑定为工具，与 DeepSeek 官方工具调用示例一致。
        if method == "function_calling" and not caps.supports_tool_choice:
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


class LocalCompatibleChatOpenAI(NormalizedChatOpenAI):
    """任意本地服务（LM Studio、vLLM、llama.cpp）的 OpenAI 兼容客户端。
    通过通用 ``openai_compatible`` 提供商接入。
    各服务的工具调用能力不同，很多不接受 LangChain 的对象形式
    ``tool_choice``。因此将 Schema 绑定为工具但不强制选择，
    提升本地结构化输出兼容性（#1057）。"""

    def with_structured_output(self, schema, *, method=None, **kwargs):
        resolved = method or get_capabilities(self.model_name).preferred_structured_method
        if resolved == "function_calling":
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """将 LangChain LLM 输入规范化为消息对象列表。

    支持原始消息列表、``ChatPromptTemplate`` 生成的 ``ChatPromptValue``；
    其他输入视为没有消息。DeepSeek 思考内容回传需要遍历历史，
    因此必须兼容直接列表调用和模板驱动的调用，否则会漏掉一部分入口。"""
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """OpenAI 兼容客户端的 DeepSeek 专属行为。

    主要处理思考模式的 reasoning_content 往返：模型返回此字段后，
    后续请求必须在对应 assistant 历史消息中再次包含它，否则 HTTP 400。
    ``_create_chat_result`` 负责接收保存，``_get_request_payload`` 负责回传。
    V4、reasoner 不支持 ``tool_choice``，这一限制由
    ``NormalizedChatOpenAI.with_structured_output`` 的能力表处理。"""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_), strict=False):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", []), strict=False
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


class MinimaxChatOpenAI(NormalizedChatOpenAI):
    """OpenAI 兼容客户端的 MiniMax 专属处理。

    M2.x 推理模型默认将 ``<think>...</think>`` 写在 message.content 中，
    可能污染保存的报告。传入 ``reasoning_split=True`` 后，
    可将思考内容分离到 ``reasoning_details``。
    必须通过 ``extra_body`` 发送，因为 OpenAI SDK 拒绝未知顶层参数（#826）。
    该标志只对 ``ModelCapabilities.requires_reasoning_split`` 为真的模型启用。
    MiniMax M2.x 仅支持字符串 "none"/"auto" 形式的 tool_choice，
    对象形式会报错，因此同样由基础客户端的能力分发去掉此参数。"""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if get_capabilities(self.model_name).requires_reasoning_split:
            # 通过 extra_body 发送参数，而不是顶层关键字。
            # OpenAI SDK（>=1.56）会校验 Completions.create 的参数，
            # 并拒绝 reasoning_split 等未知顶层字段（#826）。
            extra_body = payload.setdefault("extra_body", {})
            extra_body.setdefault("reasoning_split", True)
        return payload


# ChatGPT 订阅调用不支持的参数。
_PLAN_REFUSED = ("temperature", "top_p", "max_tokens")


class ChatGPTPlanChatOpenAI(NormalizedChatOpenAI):
    """通过 OpenAI Responses API 使用用户的 ChatGPT 订阅。

    订阅调用不允许存储响应或使用 system 角色，也拒绝部分采样及长度参数。
    请求采用流式且不保存响应，system 指令改用 developer 角色，
    不支持的参数在给出警告后移除；也不发送 API 组织或项目标识。"""

    def __init__(self, **kwargs):
        for name in _PLAN_REFUSED:
            if kwargs.pop(name, None) is not None:
                logger.warning("A ChatGPT plan does not accept %s; it is not sent.", name)
        kwargs.setdefault("streaming", True)
        kwargs.setdefault("store", False)
        super().__init__(**kwargs)
        # 登录绑定的是个人账户，API 密钥场景下的 OPENAI_ORG_ID、
        # OPENAI_PROJECT_ID 不应随订阅鉴权一起发送。
        self.openai_organization = None
        for client in (self.root_client, self.root_async_client):
            client.organization = None
            client.project = None

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for item in payload.get("input") or []:
            if isinstance(item, dict) and item.get("role") == "system":
                item["role"] = "developer"
        return payload


def _chatgpt_credential() -> Callable[[], str]:
    from .chatgpt_auth import credential

    return credential()


# 从用户配置透传给 ChatOpenAI 的关键字参数。
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "temperature", "max_tokens",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# OpenAI 的 ``reasoning_effort`` 仅适用于推理模型（如 GPT-5 及之后、o 系列）。
# 非推理模型（如 gpt-4.1、gpt-4o）会返回参数不支持的 HTTP 400；
# 因此应在调用前移除此参数。
_OPENAI_REASONING_MODEL = re.compile(r"^(?:gpt-(?:[5-9]|[1-9]\d)|o[1-9])(?:[.-]|$)")


def _supports_reasoning_effort(model: str) -> bool:
    """判断原生 OpenAI 模型是否支持 ``reasoning_effort``。"""
    return bool(_OPENAI_REASONING_MODEL.match(model.lower().strip()))


@dataclass(frozen=True)
class ProviderSpec:
    """单个 OpenAI 兼容提供商的声明式配置。

    OpenAI、xAI、DeepSeek、Qwen、GLM、MiniMax、OpenRouter、Ollama
    和用户自定义服务都使用 Chat Completions 兼容协议，
    通过配置行统一描述地址、鉴权和客户端行为。
    Anthropic、Google 使用独立 API，因此不放入此表。
    API Key 环境变量名统一放在 ``api_key_env.PROVIDER_API_KEY_ENV``，
    这里只维护提供商专属端点、密钥是否可选及协议差异。"""

    chat_class: type = NormalizedChatOpenAI   # provider quirks live in the subclass
    base_url: str | None = None            # default endpoint (None -> SDK default)
    base_url_env: str | None = None        # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False                # don't require/prompt; send a placeholder if unset
    placeholder_key: str = "EMPTY"            # sent when no key is available (keyless local servers)
    require_base_url: bool = False            # error if no base_url is resolved (generic endpoint)
    use_responses_api: bool = False           # native OpenAI Responses API
    # 对需要用户登录的提供商，``credential()`` 返回动态获取密钥的回调，
    # 每次请求可使用新 token；未登录时抛出异常。
    # 订阅 token 只能发送到固定端点，且不能与 API Key 混用。
    credential: Callable[[], Callable[[], str]] | None = None
    pinned_base_url: bool = False             # refuse any other endpoint for this provider


# OpenAI 兼容提供商的统一注册表。
# Qwen、GLM、MiniMax 等双地区提供商分别使用独立端点，
# 因为国际与中国账户的凭据不能互换（#758）。
OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "qwen-cn":    ProviderSpec(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/"),
    "glm-cn":     ProviderSpec(base_url="https://open.bigmodel.cn/api/paas/v4/"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", chat_class=MinimaxChatOpenAI),
    "minimax-cn": ProviderSpec(base_url="https://api.minimaxi.com/v1", chat_class=MinimaxChatOpenAI),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "mistral":    ProviderSpec(base_url="https://api.mistral.ai/v1"),
    "kimi":       ProviderSpec(base_url="https://api.moonshot.ai/v1"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "nvidia":     ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama",
                               chat_class=LocalCompatibleChatOpenAI),
    # 通过用户自己的登录信息使用 ChatGPT 订阅。
    "chatgpt":    ProviderSpec(base_url="https://api.openai.com/v1", chat_class=ChatGPTPlanChatOpenAI,
                               use_responses_api=True, credential=_chatgpt_credential,
                               pinned_base_url=True),
    # 通用端点由用户提供 base_url，允许省略密钥（如无鉴权本地服务）。
    "openai_compatible": ProviderSpec(
        require_base_url=True, key_optional=True, chat_class=LocalCompatibleChatOpenAI
    ),
}


# 使用账户登录，而不是手动设置 API Key 的提供商。
SIGNED_IN_PROVIDERS = tuple(key for key, spec in OPENAI_COMPATIBLE_PROVIDERS.items() if spec.credential)


def check_endpoint(provider: str, base_url: str | None) -> None:
    """固定端点的提供商若被指向其他地址，则抛出 ValueError。"""
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider.lower())
    if spec is not None and spec.pinned_base_url and base_url and base_url != spec.base_url:
        raise ValueError(f"Provider '{provider}' only connects to {spec.base_url}; "
                         "unset backend_url / TRADINGAGENTS_LLM_BACKEND_URL for it.")


def is_openai_compatible(provider: str) -> bool:
    """判断指定 ``provider`` 是否由 OpenAI 兼容注册表提供。"""
    return provider.lower() in OPENAI_COMPATIBLE_PROVIDERS


def _is_native_openai_base_url(base_url: str | None) -> bool:
    """当 ``base_url`` 未设置或指向 api.openai.com 时返回 True。

    Responses API（/v1/responses）只对原生 OpenAI 端点可用。
    如果 openai 提供商被配置为代理、网关或本地地址，
    可能只支持 Chat Completions，因此必须关闭 Responses 模式（#1024）。"""
    if not base_url:
        return True
    if "://" not in base_url:
        base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")


class OpenAIClient(BaseLLMClient):
    """OpenAI、Ollama、OpenRouter、xAI 等兼容提供商的客户端。

    原生 OpenAI 模型使用 Responses API，可结合工具和 reasoning_effort。
    第三方兼容提供商使用标准 Chat Completions 协议。"""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """根据提供商注册表返回已配置的 ChatOpenAI 实例。"""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
        chat_cls = NormalizedChatOpenAI

        if spec is not None:
            chat_cls = spec.chat_class

            # base_url 优先级：显式客户端地址（含配置或
            # TRADINGAGENTS_LLM_BACKEND_URL）高于提供商环境变量
            # （如 OLLAMA_BASE_URL），再高于提供商默认地址。
            # None 表示使用 SDK 默认值。
            env_base_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
            base_url = self.base_url or env_base_url or spec.base_url
            check_endpoint(self.provider, base_url)
            if spec.require_base_url and not base_url:
                raise ValueError(
                    f"Provider '{self.provider}' requires a base_url. Set it via "
                    "backend_url / TRADINGAGENTS_LLM_BACKEND_URL to your endpoint, "
                    "e.g. http://localhost:8000/v1 (vLLM) or http://localhost:1234/v1 "
                    "(LM Studio)."
                )
            if base_url:
                llm_kwargs["base_url"] = base_url

            # 除 key_optional 提供商外均要求 API Key，无密钥的本地服务
            # 使用占位值。环境变量名称统一在 api_key_env 中定义。
            # 订阅登录的提供商只发送自身动态获取的凭据。
            api_key_env = get_api_key_env(self.provider)
            api_key = self.kwargs.get("api_key") or (os.environ.get(api_key_env) if api_key_env else None)
            if spec.credential is not None:
                if "api_key" in self.kwargs:
                    raise ValueError(f"Provider '{self.provider}' uses its sign-in, not an api_key.")
                llm_kwargs["api_key"] = spec.credential()
            elif api_key:
                llm_kwargs["api_key"] = api_key
            elif spec.key_optional:
                llm_kwargs["api_key"] = spec.placeholder_key
            elif api_key_env:
                raise ValueError(
                    f"API key for provider '{self.provider}' is not set. "
                    f"Please set the {api_key_env} environment variable "
                    f"(e.g. add {api_key_env}=your_key to your .env file)."
                )

            # Responses API 仅对原生 OpenAI 端点启用。
            # openai 提供商若指向自定义代理、网关或本地服务，
            # 应使用 Chat Completions，不能继续开启 Responses（#1024）。
            if spec.use_responses_api and _is_native_openai_base_url(base_url):
                llm_kwargs["use_responses_api"] = True
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # 透传用户提供的关键字参数。
        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "reasoning_effort" and not _supports_reasoning_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # 根据注册表配置选择具备相应提供商兼容处理的客户端子类。
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """校验模型是否被指定提供商支持。"""
        return validate_model(self.provider, self.model)
