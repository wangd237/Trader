"""各 LLM 提供商的模型名称校验工具。"""

from .model_catalog import get_known_models

# 有些提供商的模型名由用户自行定义，例如本地服务、代理或
# 可调用大量模型的托管 OpenAI 兼容端点；对此类提供商，
# 允许任意模型字符串且不显示警告。
_ANY_MODEL_PROVIDERS = (
    "ollama", "openrouter", "openai_compatible",
    "mistral", "kimi", "groq", "nvidia", "bedrock", "chatgpt",
)

VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in _ANY_MODEL_PROVIDERS
}


def validate_model(provider: str, model: str) -> bool:
    """检查模型名称是否适用于指定提供商。

    对于 Ollama、OpenRouter、openai_compatible 等，接受任意模型名。"""
    provider_lower = provider.lower()

    if provider_lower in _ANY_MODEL_PROVIDERS:
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]
