"""通用 OpenAI 兼容提供商（vLLM、LM Studio、llama.cpp、代理）。

验证用户提供的 base_url 必须生效且不能为空，API Key 可以省略，
使用 Chat Completions 而非 Responses API，允许自定义模型 ID，
并正确执行环境变量 URL 优先级（#978）。"""

import pytest

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import LocalCompatibleChatOpenAI, OpenAIClient
from tradingagents.llm_clients.validators import validate_model


@pytest.mark.unit
def test_factory_routes_to_openai_client():
    client = create_llm_client(
        provider="openai_compatible", model="my-model", base_url="http://localhost:8000/v1"
    )
    assert isinstance(client, OpenAIClient)


@pytest.mark.unit
def test_base_url_required(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires a base_url"):
        create_llm_client(provider="openai_compatible", model="m").get_llm()


@pytest.mark.unit
def test_keyless_local_uses_placeholder_and_chat_completions(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    llm = create_llm_client(
        provider="openai_compatible", model="qwen2.5", base_url="http://localhost:8000/v1"
    ).get_llm()
    assert isinstance(llm, LocalCompatibleChatOpenAI)
    assert str(llm.openai_api_base) == "http://localhost:8000/v1"
    # 无鉴权的本地服务发送占位密钥。
    key = llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, "get_secret_value") else llm.openai_api_key
    assert key == "EMPTY"
    # 必须使用 Chat Completions，而非 OpenAI Responses API。
    assert getattr(llm, "use_responses_api", False) in (False, None)


@pytest.mark.unit
def test_optional_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-relay-123")
    llm = create_llm_client(
        provider="openai_compatible", model="m", base_url="https://relay.example/v1"
    ).get_llm()
    key = llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, "get_secret_value") else llm.openai_api_key
    assert key == "sk-relay-123"


@pytest.mark.unit
def test_any_model_accepted_no_forced_key():
    assert validate_model("openai_compatible", "literally-anything") is True
    # 保留密钥环境变量供有鉴权代理使用，但该提供商允许不配置密钥，
    # CLI 不应强制提问，保证无密钥本地服务可用。
    assert get_api_key_env("openai_compatible") == "OPENAI_COMPATIBLE_API_KEY"
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS
    assert OPENAI_COMPATIBLE_PROVIDERS["openai_compatible"].key_optional is True


@pytest.mark.unit
def test_env_backend_url_precedence():
    # #978：显式环境变量 URL 优先于菜单或默认地址。
    from cli.prompts import resolve_backend_url
    assert resolve_backend_url("openai", "https://api.openai.com/v1", env_url="http://proxy/v1") == "http://proxy/v1"
    assert resolve_backend_url("openai", "https://api.openai.com/v1", env_url=None) == "https://api.openai.com/v1"
    assert resolve_backend_url("deepseek", None, None) == "https://api.deepseek.com"


@pytest.mark.unit
def test_structured_output_suppresses_object_tool_choice(monkeypatch):
    # LM Studio、vLLM 不支持 LangChain 使用的对象形式 tool_choice（#1057）。
    # 通用提供商应绑定 Schema 为工具，但不能强制 tool_choice。
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    class Schema(BaseModel):
        x: int

    captured = {}
    monkeypatch.setattr(
        ChatOpenAI,
        "with_structured_output",
        lambda self, schema, method=None, **kw: captured.update({"method": method, **kw}) or "BOUND",
    )
    llm = create_llm_client(
        provider="openai_compatible", model="local-llm-30b", base_url="http://localhost:1234/v1"
    ).get_llm()
    out = llm.with_structured_output(Schema)
    assert out == "BOUND"
    assert captured["method"] == "function_calling"
    assert captured["tool_choice"] is None  # not the object form
