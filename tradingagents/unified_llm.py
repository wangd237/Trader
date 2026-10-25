"""Trader's three-field OpenAI-compatible API configuration.

Precedence: built-in defaults -> legacy environment overrides -> unified LLM_*.
Provider detection is based on an exact URL hostname, never on secret/key patterns.
Unrecognised endpoints use the existing OpenAI-compatible client factory.
"""

import os
from urllib.parse import urlsplit

# Only hosts represented by existing OpenAI-compatible provider clients.
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
    """True when at least one new LLM setting is nonempty."""
    return any(os.environ.get(key, "").strip() for key in
               ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"))


def resolve_unified() -> tuple[str, str, str] | None:
    """Return (provider, model, base_url), or None for legacy configuration.

    No key is put in the returned config, reports, or diagnostic messages.
    Only a loopback endpoint may omit the key.
    """
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
    # Gemini offers an OpenAI-compatible route; no native Google SDK needed.
    provider = "openai_compatible" if (
        host == "generativelanguage.googleapis.com"
    ) else _HOST_PROVIDERS.get(host, "openai_compatible")
    return provider, model, url.rstrip("/")


def apply_unified_config(config: dict) -> dict:
    """Last layer: override all LLM tiers while retaining other run settings."""
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
        # Legacy provider-specific reasoning controls cannot leak to other models.
        google_thinking_level=None,
        openai_reasoning_effort=None,
        anthropic_effort=None,
    )
    return config
