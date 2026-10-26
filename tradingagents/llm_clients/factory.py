
from typing import Any

from .base_client import BaseLLMClient


def create_llm_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    **kwargs,
) -> BaseLLMClient:
    """为指定提供商创建 LLM 客户端。
    
    按需导入提供商模块，避免仅导入工厂（例如测试收集阶段）
    就加载大型 LLM SDK，或者因为缺少 API Key 而报错。
    
    参数：
        provider：LLM 提供商名称。
        model：模型名称或标识符。
        base_url：可选的 API 端点基础地址。
        **kwargs：提供商专属的其他参数。
    
    返回：
        已配置的 BaseLLMClient 实例。
    
    异常：
        ValueError：提供商不受支持。"""
    provider_lower = provider.lower()

    # 先匹配原生（非 OpenAI）API，避免判断时不必要地导入 OpenAI 客户端；
    # 其余均视为 OpenAI 兼容接口，通过统一提供商注册表路由。
    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)

    if provider_lower == "bedrock":
        from .bedrock_client import BedrockClient
        return BedrockClient(model, base_url, **kwargs)

    from .openai_client import OpenAIClient, is_openai_compatible
    if is_openai_compatible(provider_lower):
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _coerce_max_retries(value):
    """校验 ``llm_max_retries``，确保其为非负整数。
    
    允许整数和数字字符串（环境变量以字符串形式传入）；拒绝布尔值与负数，
    让错误配置在启动时明确失败，而不是静默禁用重试。"""
    if isinstance(value, bool):
        raise ValueError(f"llm_max_retries must be an integer, not a boolean: {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"llm_max_retries must be an integer, got {value!r}") from exc
    if n < 0:
        raise ValueError(f"llm_max_retries must be >= 0, got {n}")
    return n


def _coerce_max_tokens(value):
    """校验 ``max_tokens`` 是否为正整数（环境变量的值为字符串）。"""
    if isinstance(value, bool):
        raise ValueError(f"max_tokens must be an integer, not a boolean: {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_tokens must be an integer, got {value!r}") from exc
    if n <= 0:
        raise ValueError(f"max_tokens must be > 0, got {n}")
    return n


# 支持 OpenAI ``openai_reasoning_effort`` 参数的提供商。
REASONING_EFFORT_PROVIDERS = ("openai", "chatgpt")


def build_llm_kwargs(config: dict) -> dict[str, Any]:
    """从 TradingAgents 配置中构造传给 ``create_llm_client`` 的关键字参数。"""
    kwargs = {}
    provider = config.get("llm_provider", "").lower()

    if provider == "google":
        thinking_level = config.get("google_thinking_level")
        if thinking_level:
            kwargs["thinking_level"] = thinking_level

    elif provider in REASONING_EFFORT_PROVIDERS:
        reasoning_effort = config.get("openai_reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

    elif provider == "anthropic":
        effort = config.get("anthropic_effort")
        if effort:
            kwargs["effort"] = effort

    # temperature 是跨提供商参数，设置后统一透传。
    # 这里调用 float()，确保环境变量 TRADINGAGENTS_TEMPERATURE
    # 中的字符串（如 "0.2"）与代码传入的浮点值表现一致。
    temperature = config.get("temperature")
    if temperature is not None and temperature != "":
        kwargs["temperature"] = float(temperature)

    # SDK 重试次数为跨提供商设置；仅在显式设置时透传，
    # 否则保留各提供商自己的默认值（通常为 2，#1091）。
    max_retries = config.get("llm_max_retries")
    if max_retries is not None and max_retries != "":
        kwargs["max_retries"] = _coerce_max_retries(max_retries)

    # 输出 token 上限同样跨提供商，但 Gemini 使用
    # ``max_output_tokens`` 参数名；需要按提供商转换后传入（#1204）。
    max_tokens = config.get("max_tokens")
    if max_tokens is not None and max_tokens != "":
        key = "max_output_tokens" if provider == "google" else "max_tokens"
        kwargs[key] = _coerce_max_tokens(max_tokens)

    return kwargs


def tier_provider(config: dict, tier: str) -> str:
    """确定 ``quick`` 或 ``deep`` 模型层级使用的提供商：优先使用该层级指定的值，否则使用 ``llm_provider``。"""
    return (config.get(f"{tier}_think_provider") or config["llm_provider"]).lower()


def create_tier_client(config: dict, tier: str, **extra) -> BaseLLMClient:
    """创建 ``quick`` 或 ``deep`` 模型层级对应的客户端（#1440）。
    
    若层级使用独立提供商，则使用该提供商的推理参数、输出 token 上限
    参数名和专属端点；未配置独立端点时使用该提供商的默认值。
    ``backend_url`` 只属于全局的 ``llm_provider``。"""
    provider = tier_provider(config, tier)
    if provider == config["llm_provider"].lower():
        base_url = config.get(f"{tier}_think_backend_url") or config.get("backend_url")
    else:
        base_url = config.get(f"{tier}_think_backend_url")
    kwargs = build_llm_kwargs({**config, "llm_provider": provider})
    if config.get("unified_llm"):
        # 凭据不能进入运行配置、报告或检查点指纹。
        import os
        kwargs["api_key"] = os.environ.get("LLM_API_KEY") or "EMPTY"
    return create_llm_client(provider=provider, model=config[f"{tier}_think_llm"], base_url=base_url,
                             **kwargs, **extra)
