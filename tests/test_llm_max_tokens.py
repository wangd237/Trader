"""可配置的输出 token 上限（#1204）。

部分模型或网关组合的推理输出会非常长，可能挂起或达到空闲超时。
配置项 max_tokens 会传递给所有提供商；
Gemini 对应字段名是 max_output_tokens，需要转换。"""
from __future__ import annotations

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.llm_clients.factory import _coerce_max_tokens, build_llm_kwargs

# --- 类型转换与校验 -------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [(1, 1), (8192, 8192), ("4096", 4096)])
def test_coerce_accepts_positive_ints_and_numeric_strings(value, expected):
    assert _coerce_max_tokens(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, "0", "-5"])
def test_coerce_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="> 0"):
        _coerce_max_tokens(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, False])
def test_coerce_rejects_booleans(bad):
    with pytest.raises(ValueError, match="boolean"):
        _coerce_max_tokens(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["abc", "1.5", None])
def test_coerce_rejects_non_integers(bad):
    with pytest.raises(ValueError, match="integer"):
        _coerce_max_tokens(bad)


# --- 按提供商正确字段名透传 --------------


@pytest.mark.unit
def test_not_forwarded_when_unset():
    kwargs = build_llm_kwargs({"llm_provider": "openai", "max_tokens": None})
    assert "max_tokens" not in kwargs
    assert "max_output_tokens" not in kwargs


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "anthropic", "deepseek", "openai_compatible"])
def test_forwarded_as_max_tokens_for_non_google(provider):
    kwargs = build_llm_kwargs({"llm_provider": provider, "max_tokens": 8192})
    assert kwargs["max_tokens"] == 8192
    assert "max_output_tokens" not in kwargs


@pytest.mark.unit
def test_forwarded_as_max_output_tokens_for_google():
    # Gemini 的参数名称不同，直接透传 max_tokens 会被拒绝。
    kwargs = build_llm_kwargs({"llm_provider": "google", "max_tokens": 8192})
    assert kwargs["max_output_tokens"] == 8192
    assert "max_tokens" not in kwargs


@pytest.mark.unit
def test_env_string_is_coerced():
    kwargs = build_llm_kwargs({"llm_provider": "openai", "max_tokens": "4096"})
    assert kwargs["max_tokens"] == 4096


@pytest.mark.unit
def test_invalid_value_fails_loudly():
    with pytest.raises(ValueError):
        build_llm_kwargs({"llm_provider": "openai", "max_tokens": 0})


# --- 客户端允许透传字段 --------------------------------

@pytest.mark.unit
def test_openai_and_google_clients_accept_the_kwarg():
    from tradingagents.llm_clients import openai_client
    from tradingagents.llm_clients.google_client import GoogleClient  # noqa: F401
    assert "max_tokens" in openai_client._PASSTHROUGH_KWARGS
    # Google 客户端在实例构造时传递 max_output_tokens。
    llm = GoogleClient("gemini-3.5-flash", api_key="x", max_output_tokens=8192).get_llm()
    assert getattr(llm, "max_output_tokens", None) == 8192


# --- 环境变量覆盖 -----------------------------------------------------------

def _config_with_env(monkeypatch, **overrides):
    """按当前测试环境变量构造默认配置。"""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return default_config_module.build_default_config()


@pytest.mark.unit
def test_default_is_none(monkeypatch):
    config = _config_with_env(monkeypatch)
    assert config["max_tokens"] is None


@pytest.mark.unit
def test_env_override_sets_config(monkeypatch):
    config = _config_with_env(monkeypatch, TRADINGAGENTS_MAX_TOKENS="8192")
    assert config["max_tokens"] == "8192"
    assert _coerce_max_tokens(config["max_tokens"]) == 8192
