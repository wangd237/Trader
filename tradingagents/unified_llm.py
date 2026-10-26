"""Trader 统一的 OpenAI 兼容 API 配置，包含三个核心字段。

优先级：内置默认值 → 旧版环境变量覆盖 → 统一的 LLM_* 配置。
根据 URL 的完整主机名识别提供商，绝不通过密钥特征推断。
无法识别的端点交给现有 OpenAI 兼容客户端工厂处理。"""

import os
from urllib.parse import urlsplit

# 只识别已有 OpenAI 兼容客户端支持的主机。
_HOST_PROVIDERS = {
    "api.openai.com": "openai",
    "api.deepseek.com": "deepseek",
    "api.x.ai": "xai",
    "dashscope-intl.aliyuncs.com": "qwen",
    "dashscope.aliyuncs.com": "qwen-cn",
    "api.z.ai": "glm",
    "open.bigmodel.cn": "glm-cn",
    "api.minimax.io": "minimax",
    "api.minimaxi.com": "minimax-cn",
    "openrouter.ai": "openrouter",
    "api.mistral.ai": "mistral",
    "api.moonshot.ai": "kimi",
    "api.groq.com": "groq",
    "integrate.api.nvidia.com": "nvidia",
}
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def unified_requested() -> bool:
    """任意一项新的 LLM 配置非空时返回 True。"""
    return any(os.environ.get(key, "").strip() for key in
               ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"))


def resolve_unified() -> tuple[str, str, str] | None:
    """返回 (provider, model, base_url)；使用旧版配置时返回 None。

    API Key 不会写入配置返回值、报告或诊断消息。
    只有本机回环地址允许省略密钥。"""
    if not unified_requested():
        return None
    model = os.environ.get("LLM_MODEL", "").strip()
    url = os.environ.get("LLM_BASE_URL", "").strip()
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not model or not url:
        raise ValueError("Unified LLM configuration requires LLM_MODEL and LLM_BASE_URL")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("LLM_BASE_URL must be an absolute http(s) API base URL without credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM_BASE_URL must not contain a query string or fragment")
    host = parsed.hostname.lower()
    is_local = host in _LOCAL_HOSTS
    if parsed.scheme != "https" and not is_local:
        raise ValueError("LLM_BASE_URL must use HTTPS unless the host is localhost")
    if not key and not is_local:
        raise ValueError("LLM_API_KEY is required for non-local LLM_BASE_URL")
    if host == "api.anthropic.com" or (
        host == "generativelanguage.googleapis.com" and "/openai" not in parsed.path.lower()
    ):
        raise ValueError("LLM_BASE_URL points to a native API. Supply its OpenAI-compatible endpoint instead")
    # Gemini 支持 OpenAI 兼容端点，因此无需使用原生 Google SDK。
    provider = "openai_compatible" if (
        host == "generativelanguage.googleapis.com"
    ) else _HOST_PROVIDERS.get(host, "openai_compatible")
    return provider, model, url.rstrip("/")


def apply_unified_config(config: dict) -> dict:
    """配置合并的最后一层：覆盖全部 LLM 模型层级，保留其他运行配置。"""
    resolved = resolve_unified()
    if resolved is None:
        config["unified_llm"] = False
        return config
    provider, model, url = resolved
    config.update(
        unified_llm=True,
        llm_provider=provider,
        backend_url=url,
        quick_think_llm=model,
        deep_think_llm=model,
        quick_think_provider=None,
        deep_think_provider=None,
        quick_think_backend_url=None,
        deep_think_backend_url=None,
        # 旧版提供商专属推理参数不能泄漏给其他模型。
        google_thinking_level=None,
        openai_reasoning_effort=None,
        anthropic_effort=None,
    )
    return config
