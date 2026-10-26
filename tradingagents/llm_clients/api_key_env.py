"""提供商与 API Key 环境变量的统一映射。

集中定义各 LLM 提供商使用的密钥环境变量名，供 CLI 交互式密钥输入
（cli/prompts.ensure_api_key）以及其他鉴权检查使用。
新增提供商时只需在这里登记，CLI 就能在调用 API 前自动提示输入，
而不是等首次请求失败。"""

from __future__ import annotations

PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "azure":      "AZURE_OPENAI_API_KEY",
    # Bedrock 使用 AWS 凭证链鉴权，而不是单个密钥环境变量。
    "bedrock":    None,
    "xai":        "XAI_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    # 双地区提供商分别使用自己的账户，国际端点与中国端点
    # 的密钥不能互换。
    "qwen":       "DASHSCOPE_API_KEY",
    "qwen-cn":    "DASHSCOPE_CN_API_KEY",
    "glm":        "ZHIPU_API_KEY",
    "glm-cn":     "ZHIPU_CN_API_KEY",
    "minimax":    "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # 其他托管的 OpenAI 兼容提供商，模型由用户指定。
    # kimi 对应 Moonshot AI，nvidia 对应 NVIDIA NIM。
    "mistral":    "MISTRAL_API_KEY",
    "kimi":       "MOONSHOT_API_KEY",
    "groq":       "GROQ_API_KEY",
    "nvidia":     "NVIDIA_API_KEY",
    # ChatGPT 订阅通过用户登录鉴权，而不是 API Key。
    "chatgpt":    None,
    # 本地运行时不需要鉴权。
    "ollama":     None,
    # 通用 OpenAI 兼容端点可在设置了密钥时使用它（例如带认证的中继），
    # 但提供商注册表中标记为可选，因此 CLI 不会强制提问，
    # 无密钥的本地服务仍可使用。
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}


def get_api_key_env(provider: str) -> str | None:
    """返回指定 `provider` 对应的 API Key 环境变量名；不适用时返回 None。

    未知提供商也返回 None，调用者应理解为“无法检查密钥”，
    而不是“无需密钥”。"""
    return PROVIDER_API_KEY_ENV.get(provider.lower())
