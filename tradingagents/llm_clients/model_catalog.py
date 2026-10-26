"""CLI 模型选择与合法性校验共用的模型目录。"""

from __future__ import annotations

ModelOption = tuple[str, str]
ProviderModeOptions = dict[str, dict[str, list[ModelOption]]]

# 对模型数量多且变化频繁的提供商，仅显示“自定义模型 ID”，
# 避免维护很快过时的固定列表。
_CUSTOM_ONLY: dict[str, list[ModelOption]] = {
    "quick": [("Custom model ID", "custom")],
    "deep": [("Custom model ID", "custom")],
}


# Z.AI（国际）与 BigModel（中国）的 GLM 共用模型列表。
# 来源：docs.z.ai 的 GLM Coding Plan 和 LLM 文档。
# GLM 4.7 及更新版本可通过 thinking={"type":"enabled"} 启用思考模式。
_GLM_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("GLM-5.3-Flash - Fast, cost-efficient, 1M ctx", "glm-5.3-flash"),
        ("GLM-5-Turbo - Fast, switchable thinking modes", "glm-5-turbo"),
        ("GLM-4.5-Air - Lightweight, cost-efficient", "glm-4.5-air"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("GLM-5.3 - Latest flagship, 1M ctx", "glm-5.3"),
        ("GLM-5.2 - 744B, 1M ctx", "glm-5.2"),
        ("GLM-5.1 - 745B, 200K ctx", "glm-5.1"),
        ("GLM-4.7 - Previous-gen flagship", "glm-4.7"),
        ("Custom model ID", "custom"),
    ],
}


# Qwen 国际（dashscope-intl）与中国（dashscope）端点的共用模型列表。
# 来源：阿里云百炼 Model Studio 的旗舰模型及性价比模型列表。
#
# 下拉菜单只展示明确版本的模型 ID。没有版本号的 qwen-plus、qwen-flash
# 等别名会自动升级，阿里云可能随时替换其底层模型，导致行为变化。
# 希望固定模型代际的用户应选择带版本号的 ID；
# 希望自动跟随最新版的用户可输入“自定义模型 ID”。
_QWEN_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("Qwen 3.8 Flash - Latest fast model, 1M ctx", "qwen3.8-flash"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Qwen 3.8 Max - Latest flagship", "qwen3.8-max"),
        ("Qwen 3.8 Flash - Fast alternative, 1M ctx", "qwen3.8-flash"),
        ("Custom model ID", "custom"),
    ],
}


# MiniMax 国际与中国端点使用相同模型 ID。
# 完整列表来源：platform.minimax.io/docs/api-reference/text-openai-api。
# M3 上下文窗口为 100 万 token，M2.x 系列为 204,800 token。
#
# Kimi（Moonshot）模型参考 platform.kimi.ai/docs/models；
# “自定义模型 ID”可用于列表尚未收录的新模型。
# k2.7-code 等代码专用模型不纳入分析模型推荐列表。
_KIMI_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("Kimi K2.6 - 256K ctx, thinking modes, agent tasks", "kimi-k2.6"),
        ("Kimi K3 - Flagship, 1M ctx", "kimi-k3"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Kimi K3 - Flagship, 1M ctx, native visual understanding", "kimi-k3"),
        ("Kimi K2.6 - 256K ctx, thinking modes, agent tasks", "kimi-k2.6"),
        ("Custom model ID", "custom"),
    ],
}


_MINIMAX_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("MiniMax-M3 - Latest, 1M ctx, native multimodal", "MiniMax-M3"),
        ("MiniMax-M2.7-highspeed - Fast M2.7, 204K ctx, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5-highspeed - Previous-gen highspeed, 204K ctx", "MiniMax-M2.5-highspeed"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("MiniMax-M3 - Latest flagship, 1M ctx, multimodal coding/agent", "MiniMax-M3"),
        ("MiniMax-M2.7 - Previous flagship, 204K ctx", "MiniMax-M2.7"),
        ("MiniMax-M2.7-highspeed - Same quality as M2.7, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5 - Earlier flagship, 204K ctx", "MiniMax-M2.5"),
        ("Custom model ID", "custom"),
    ],
}


MODEL_OPTIONS: ProviderModeOptions = {
    "openai": {
        "quick": [
            ("GPT-6 Luna - Fast, high-volume and cost-efficient", "gpt-6-luna"),
            ("GPT-5.6 Luna - Previous-gen fast tier", "gpt-5.6-luna"),
            ("GPT-5.6 Terra - Balances intelligence and cost", "gpt-5.6-terra"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GPT-6 Sol - Complex reasoning and analysis", "gpt-6-sol"),
            ("GPT-6 Astra - Hardest multi-step reasoning", "gpt-6-astra"),
            ("GPT-5.6 - Previous-gen reasoning (Sol)", "gpt-5.6"),
            ("GPT-5.5 - Earlier frontier, 1M context", "gpt-5.5"),
            ("Custom model ID", "custom"),
        ],
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 5 - Best speed and intelligence balance", "claude-sonnet-5"),
            ("Claude Haiku 4.5 - Fastest with near-frontier intelligence", "claude-haiku-4-5"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Claude Opus 5.5 - Frontier agentic and enterprise work", "claude-opus-5-5"),
            ("Claude Fable 5.1 - Most capable, demanding long-horizon reasoning", "claude-fable-5-1"),
            ("Claude Sonnet 5 - Near-frontier intelligence at Sonnet cost", "claude-sonnet-5"),
            ("Custom model ID", "custom"),
        ],
    },
    "google": {
        "quick": [
            ("Gemini 3.8 Flash - Most capable Flash", "gemini-3.8-flash"),
            ("Gemini 3.5 Flash Lite - Fast and cost-efficient", "gemini-3.5-flash-lite"),
            ("Gemini 3.1 Flash Lite - Most cost-efficient", "gemini-3.1-flash-lite"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Gemini 3.8 Flash - Most capable Flash, 1M context", "gemini-3.8-flash"),
            ("Gemini 3.1 Pro - Reasoning-first, complex workflows (preview)", "gemini-3.1-pro-preview"),
            ("Gemini 3.5 Flash - Previous Flash, strong agentic + coding", "gemini-3.5-flash"),
            ("Custom model ID", "custom"),
        ],
    },
    "xai": {
        "quick": [
            ("Grok 4.6 - Latest flagship, fastest, 500K ctx", "grok-4.6"),
            ("Grok Build 0.1 - Coding-specialized, 256K ctx", "grok-build-0.1"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Grok 4.6 - Latest flagship, 500K ctx", "grok-4.6"),
            ("Grok 4.5 - Previous flagship, coding and agentic", "grok-4.5"),
            ("Grok 4.3 - Older generation, 1M ctx", "grok-4.3"),
            ("Custom model ID", "custom"),
        ],
    },
    # DeepSeek 的 deepseek-chat、deepseek-reasoner 别名已弃用
    # （2026-07-24），当前映射到 V4 Flash，因此直接展示 V4 ID。
    # V4 Flash 同时支持非思考与思考模式；客户端负责回传 reasoning_content。
    "deepseek": {
        "quick": [
            ("DeepSeek Flash - V4.1 Flash, fast, 1M ctx", "deepseek-flash"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V4 Pro - Flagship", "deepseek-v4-pro"),
            ("DeepSeek Flash - V4.1 Flash, fast, 1M ctx", "deepseek-flash"),
            ("Custom model ID", "custom"),
        ],
    },
    # Qwen 国际与中国端点使用相同模型 ID，两个提供商配置共用列表。
    "qwen": _QWEN_MODELS,
    "qwen-cn": _QWEN_MODELS,
    # Z.AI 国际端点和 BigModel 中国端点提供相同模型 ID，
    # 两个提供商配置共用列表。
    "glm": _GLM_MODELS,
    "glm-cn": _GLM_MODELS,
    # MiniMax 国际（.io）与中国（.com）端点使用相同模型 ID，
    # 因此两个提供商配置共用列表。
    "kimi": _KIMI_MODELS,
    "minimax": _MINIMAX_MODELS,
    "minimax-cn": _MINIMAX_MODELS,
    # OpenRouter 模型列表动态获取；Azure 接受任意已部署模型名。
    # Ollama 模型标签不再强调“本地”，因为端点可通过 OLLAMA_BASE_URL
    # 配置，既可以连接 localhost 也能连接远程服务。
    # 实际端点在提供商选择后由 cli.prompts.confirm_ollama_endpoint()
    # 单独提示。用户还可以输入“自定义模型 ID”，使用通过
    # `ollama pull` 安装的其他模型。
    "ollama": {
        "quick": [
            ("Qwen3:latest (8B)", "qwen3:latest"),
            ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"),
            ("GPT-OSS:latest (20B)", "gpt-oss:latest"),
            ("Qwen3:latest (8B)", "qwen3:latest"),
            ("Custom model ID", "custom"),
        ],
    },
    # 通用 OpenAI 兼容端点由用户自己的服务器决定支持的模型，
    # 因此只提供“自定义模型 ID”选项。
    "openai_compatible": _CUSTOM_ONLY,
    # 对大量托管的 OpenAI 兼容模型提供商，仅提供“自定义模型 ID”，
    # 避免模型频繁更新导致列表过时。端点和密钥由提供商适配层配置，
    # 用户输入自身账户有权限访问的模型。
    "mistral": {
        "quick": [
            ("Mistral Small 4 - Fast, 262K ctx", "mistral-small-2603"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Mistral Medium 3.5 - 262K ctx", "mistral-medium-2604"),
            ("Mistral Small 4 - Fast, 262K ctx", "mistral-small-2603"),
            ("Custom model ID", "custom"),
        ],
    },
    "groq": _CUSTOM_ONLY,
    "nvidia": _CUSTOM_ONLY,
    # Bedrock 模型 ID 和跨地区推理配置 ID 由用户指定。
    "bedrock": _CUSTOM_ONLY,
    # 完成登录后从 OpenAI 实时读取订阅可用的模型列表。
    "chatgpt": _CUSTOM_ONLY,
}


def get_model_options(provider: str, mode: str) -> list[ModelOption]:
    """返回指定提供商与选择模式的共享模型选项。"""
    return MODEL_OPTIONS[provider.lower()][mode]


# 有些模型仍由提供商提供，但不会出现在菜单中，
# 例如已从菜单移除的旧模型，或在界面上显示了更短名称的正式 ID。
# 校验仍需识别这些模型，以免已有配置产生未知模型警告。
LEGACY_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.6-sol"],
    "xai": ["grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning",
            "grok-4.20-multi-agent-0309"],
    "deepseek": ["deepseek-v4-flash"],
    "qwen": ["qwen3.7-max", "qwen3.7-plus", "qwen3.6-max", "qwen3.6-plus"],
    "anthropic": ["claude-opus-5", "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7"],
}


def get_known_models() -> dict[str, list[str]]:
    """合并 CLI 共用目录与历史模型 ID，生成已知模型名称集合。"""
    return {
        provider: sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
            | set(LEGACY_MODELS.get(provider, []))
        )
        for provider, mode_options in MODEL_OPTIONS.items()
    }
