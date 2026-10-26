"""测试 OLLAMA_BASE_URL 是否同时对 CLI 和客户端生效。"""

from __future__ import annotations

import re

import pytest

# Rich 会为数字和 URL 添加颜色转义序列，拆开待断言字符串；
# 应去除 ANSI 转义码，避免测试依赖终端环境。
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _console_out(capsys) -> str:
    return _ANSI.sub("", capsys.readouterr().out)


# ---- openai_client side: registry-driven base_url resolution --------------


def _client_module():
    import tradingagents.llm_clients.openai_client as mod
    return mod


def _base_url(mod, provider, **kwargs):
    return str(mod.OpenAIClient(model="m", provider=provider, **kwargs).get_llm().openai_api_base)


def test_resolver_returns_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    mod = _client_module()
    assert _base_url(mod, "ollama") == "http://localhost:11434/v1"


def test_resolver_returns_env_when_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434/v1")
    mod = _client_module()
    assert _base_url(mod, "ollama") == "http://remote-ollama:11434/v1"


def test_resolver_evaluation_is_call_time(monkeypatch):
    """模块已导入后再设置环境变量，也必须正确生效。"""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    mod = _client_module()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://late-set:11434/v1")
    assert _base_url(mod, "ollama") == "http://late-set:11434/v1"


def test_resolver_does_not_affect_other_providers(monkeypatch):
    """OLLAMA_BASE_URL 不能影响 xAI、DeepSeek 等其他提供商。"""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://elsewhere/v1")
    mod = _client_module()
    assert _base_url(mod, "xai") == "https://api.x.ai/v1"
    assert _base_url(mod, "deepseek") == "https://api.deepseek.com"


def test_client_get_llm_picks_up_env(monkeypatch):
    """端到端检查 OllamaClient.get_llm() 遵循 OLLAMA_BASE_URL。"""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-ollama:11434/v1")
    mod = _client_module()
    client = mod.OpenAIClient(model="llama3.1", provider="ollama")
    llm = client.get_llm()
    assert "my-ollama" in str(llm.openai_api_base)


def test_explicit_base_url_overrides_env(monkeypatch):
    """显式传入的客户端 base_url 应优先于环境变量。"""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-set:11434/v1")
    mod = _client_module()
    client = mod.OpenAIClient(
        model="llama3.1",
        provider="ollama",
        base_url="http://explicit:11434/v1",
    )
    llm = client.get_llm()
    assert "explicit" in str(llm.openai_api_base)
    assert "env-set" not in str(llm.openai_api_base)


# ---- confirm_ollama_endpoint UX -------------------------------------------


def test_confirm_endpoint_shows_default(monkeypatch, capsys):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://localhost:11434/v1")
    out = _console_out(capsys)
    assert "http://localhost:11434/v1" in out
    assert "OLLAMA_BASE_URL" not in out  # not from env
    assert "Note" not in out  # no warnings for the canonical default


def test_confirm_endpoint_marks_env_origin(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host:11434/v1")
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://remote-host:11434/v1")
    out = _console_out(capsys)
    assert "http://remote-host:11434/v1" in out
    assert "OLLAMA_BASE_URL" in out


def test_confirm_endpoint_warns_on_missing_scheme(monkeypatch, capsys):
    """用户设置 OLLAMA_BASE_URL=0.0.0.128 时，应提示正确 URL 格式。"""
    monkeypatch.setenv("OLLAMA_BASE_URL", "0.0.0.128")
    from cli import prompts
    prompts.confirm_ollama_endpoint("0.0.0.128")
    out = _console_out(capsys)
    assert "missing a scheme" in out
    assert "http://<host>:11434/v1" in out


def test_confirm_endpoint_warns_on_non_default_port_remote(monkeypatch, capsys):
    """远程主机未指定 :11434 端口时，应给出非强制的提示。"""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host/v1")
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://remote-host/v1")
    out = _console_out(capsys)
    assert "port 11434" in out


def test_confirm_endpoint_quiet_on_local_no_port(monkeypatch, capsys):
    """本地主机缺少端口时，不应触发远程端口不匹配提示。"""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost/v1")
    from cli import prompts
    prompts.confirm_ollama_endpoint("http://localhost/v1")
    out = _console_out(capsys)
    assert "Note" not in out  # localhost is fine without explicit port


def test_ollama_model_labels_no_local_suffix():
    """端点可以连接远程服务，选项文案不能固定声明为“(local)”。"""
    from tradingagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        labels = [label for label, _ in get_model_options("ollama", mode)]
        assert all("local" not in label for label in labels), labels


def test_ollama_offers_custom_model_id():
    """自行拉取模型的 Ollama 用户可以选择“Custom model ID”。"""
    from tradingagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        entries = get_model_options("ollama", mode)
        values = [v for _, v in entries]
        assert "custom" in values, f"Ollama {mode!r} missing 'custom' option: {entries}"
        # 自定义选项应放在列表最后，避免挤掉预设模型。
        assert values[-1] == "custom", f"'custom' should be last entry: {values}"


@pytest.mark.unit
def test_structured_output_suppresses_object_tool_choice(monkeypatch):
    """Ollama 与其他本地服务一样拒绝对象形式的 tool_choice（#1062）；
    本地模型 ID 未必有专属能力表，因此应使用通用的本地兼容客户端。"""
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    from tradingagents.llm_clients import create_llm_client

    class Schema(BaseModel):
        x: int

    captured = {}
    monkeypatch.setattr(
        ChatOpenAI,
        "with_structured_output",
        lambda self, schema, method=None, **kw: captured.update({"method": method, **kw}) or "BOUND",
    )

    create_llm_client(provider="ollama", model="qwen3:30b").get_llm().with_structured_output(Schema)

    assert captured["tool_choice"] is None
