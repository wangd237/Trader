"""只对支持推理的 OpenAI 模型启用 ``reasoning_effort``。

gpt-4.1、gpt-4o 等非推理模型不支持该参数，会返回 HTTP 400；
客户端必须移除它，而 GPT-5 和 o 系列可以使用。"""

import pytest

from tradingagents.llm_clients.openai_client import (
    OpenAIClient,
    _supports_reasoning_effort,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.5", True), ("gpt-5.4", True), ("gpt-5.4-mini", True),
        ("gpt-5.5-pro", True), ("gpt-6-astra", True), ("o1", True), ("o3-mini", True),
        ("gpt-4.1", False), ("gpt-4o", False), ("gpt-4o-mini", False),
        ("gpt-3.5-turbo", False), ("gpt-10", True),
        ("gpt-5foo", False), ("gpt-60x", False), ("o3rd-party", False),
    ],
)
def test_supports_reasoning_effort(model, expected):
    assert _supports_reasoning_effort(model) is expected


def _effort_on(model, monkeypatch):
    # 使用模拟密钥即可让 get_llm() 构造客户端，无须实际联网。
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAIClient(model, provider="openai", reasoning_effort="low").get_llm()
    return getattr(llm, "reasoning_effort", None)


def test_reasoning_model_receives_effort(monkeypatch):
    assert _effort_on("gpt-5.4-mini", monkeypatch) == "low"


def test_non_reasoning_model_drops_effort(monkeypatch):
    # gpt-4.1 收到 reasoning_effort 会返回 400，因此必须移除。
    assert _effort_on("gpt-4.1", monkeypatch) is None
