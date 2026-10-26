"""可配置的 LLM SDK 重试预算（#1090/#1091）。

临时 HTTP 429 限流可能使多 Agent 任务失败，而 SDK 默认只重试两次。
新增可选 llm_max_retries 并传递给所有提供商客户端。"""
from __future__ import annotations

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.llm_clients.factory import _coerce_max_retries, build_llm_kwargs

# --- 类型转换与校验 -------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [(0, 0), (2, 2), (10, 10), ("6", 6)])
def test_coerce_accepts_non_negative_ints_and_numeric_strings(value, expected):
    assert _coerce_max_retries(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, "-3"])
def test_coerce_rejects_negative(bad):
    with pytest.raises(ValueError, match=">= 0"):
        _coerce_max_retries(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, False])
def test_coerce_rejects_booleans(bad):
    with pytest.raises(ValueError, match="boolean"):
        _coerce_max_retries(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["abc", "1.5", None])
def test_coerce_rejects_non_integers(bad):
    with pytest.raises(ValueError, match="integer"):
        _coerce_max_retries(bad)


# --- 提供商参数透传 --------------------------------------


@pytest.mark.unit
def test_not_forwarded_when_unset():
    kwargs = build_llm_kwargs({"llm_provider": "openai", "llm_max_retries": None})
    assert "max_retries" not in kwargs


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_forwarded_across_providers(provider):
    kwargs = build_llm_kwargs({"llm_provider": provider, "llm_max_retries": 6})
    assert kwargs["max_retries"] == 6


@pytest.mark.unit
def test_forwarded_env_string_is_coerced():
    # 环境变量最初是字符串，由下游像 temperature 一样进行类型转换。
    kwargs = build_llm_kwargs({"llm_provider": "openai", "llm_max_retries": "4"})
    assert kwargs["max_retries"] == 4


@pytest.mark.unit
def test_invalid_config_value_fails_loudly():
    with pytest.raises(ValueError):
        build_llm_kwargs({"llm_provider": "openai", "llm_max_retries": -1})


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
    assert config["llm_max_retries"] is None


@pytest.mark.unit
def test_env_override_sets_config(monkeypatch):
    config = _config_with_env(monkeypatch, TRADINGAGENTS_LLM_MAX_RETRIES="8")
    # 默认值为 None 的配置项也由下游将环境变量字符串转换为目标类型。
    assert config["llm_max_retries"] == "8"
    assert _coerce_max_retries(config["llm_max_retries"]) == 8
