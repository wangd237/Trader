"""统一 Trader LLM 配置测试：优先级、安全边界和工厂路由。"""

import pytest

from tradingagents.default_config import build_default_config
from tradingagents.llm_clients import factory
from tradingagents.unified_llm import apply_unified_config, resolve_unified, unified_requested

KEYS = (
    "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL",
    "TRADINGAGENTS_LLM_PROVIDER", "TRADINGAGENTS_QUICK_THINK_LLM",
    "TRADINGAGENTS_DEEP_THINK_LLM", "TRADINGAGENTS_LLM_BACKEND_URL",
    "TRADINGAGENTS_QUICK_THINK_PROVIDER", "TRADINGAGENTS_DEEP_THINK_PROVIDER",
)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def set_three(monkeypatch, url="https://api.deepseek.com/v1", model="deepseek-chat", key="test-key"):
    monkeypatch.setenv("LLM_API_KEY", key)
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv("LLM_BASE_URL", url)


@pytest.mark.unit
@pytest.mark.parametrize(("url", "provider"), [
    ("https://api.openai.com/v1", "openai"),
    ("https://api.deepseek.com/v1", "deepseek"),
    ("https://openrouter.ai/api/v1", "openrouter"),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-cn"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "openai_compatible"),
    ("https://llm.example.com/v1", "openai_compatible"),
    ("http://localhost:11434/v1", "openai_compatible"),
])
def test_detect_provider_by_endpoint(monkeypatch, url, provider):
    set_three(monkeypatch, url)
    assert resolve_unified() == (provider, "deepseek-chat", url)


@pytest.mark.unit
def test_unified_overrides_legacy_tiers_but_not_unrelated_settings(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "google")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "old")
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_PROVIDER", "anthropic")
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "3")
    set_three(monkeypatch, model="deepseek-reasoner")
    cfg = build_default_config()
    assert cfg["unified_llm"] is True
    assert cfg["llm_provider"] == "deepseek"
    assert cfg["quick_think_llm"] == cfg["deep_think_llm"] == "deepseek-reasoner"
    assert cfg["quick_think_provider"] is cfg["deep_think_provider"] is None
    assert cfg["max_debate_rounds"] == 3
    assert "test-key" not in str(cfg)


@pytest.mark.unit
def test_factory_passes_single_key_without_provider_specific_env(monkeypatch):
    set_three(monkeypatch)
    cfg = build_default_config()
    seen = []

    def make(**kwargs):
        seen.append(kwargs)
        return kwargs

    monkeypatch.setattr(factory, "create_llm_client", make)
    for tier in ("quick", "deep"):
        factory.create_tier_client(cfg, tier)
    assert len(seen) == 2
    assert all(x["provider"] == "deepseek" and x["model"] == "deepseek-chat" for x in seen)
    assert all(x["api_key"] == "test-key" for x in seen)


@pytest.mark.unit
def test_openai_client_accepts_unified_api_key_without_provider_key(monkeypatch):
    set_three(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = build_default_config()
    llm = factory.create_tier_client(cfg, "quick").get_llm()
    assert llm.model_name == "deepseek-chat"
    assert "api.deepseek.com" in str(llm.openai_api_base)


@pytest.mark.unit
@pytest.mark.parametrize("url", [
    "https://api.deepseek.com/v1?token=secret",
    "http://api.deepseek.com/v1",
    "https://user:password@api.deepseek.com/v1",
    "ftp://example.org/v1",
    "https://api.anthropic.com/v1",
    "https://generativelanguage.googleapis.com/v1beta",
])
def test_reject_invalid_or_native_endpoints(monkeypatch, url):
    set_three(monkeypatch, url)
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        resolve_unified()


@pytest.mark.unit
def test_reject_missing_remote_key(monkeypatch):
    set_three(monkeypatch, key="")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        resolve_unified()


@pytest.mark.unit
def test_keyless_loopback_is_supported(monkeypatch):
    set_three(monkeypatch, url="http://127.0.0.1:1234/v1", key="")
    assert resolve_unified()[0] == "openai_compatible"
    assert build_default_config()["unified_llm"]


@pytest.mark.unit
def test_incomplete_configuration_fails_early(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    assert unified_requested()
    with pytest.raises(ValueError, match="LLM_MODEL and LLM_BASE_URL"):
        build_default_config()


@pytest.mark.unit
def test_legacy_mode_is_preserved():
    assert not unified_requested()
    cfg = apply_unified_config({"llm_provider": "anthropic"})
    assert cfg == {"llm_provider": "anthropic", "unified_llm": False}


@pytest.mark.unit
def test_fake_host_not_mistaken_for_official(monkeypatch):
    set_three(monkeypatch, "https://api.deepseek.com.attacker.test/v1")
    assert resolve_unified()[0] == "openai_compatible"
