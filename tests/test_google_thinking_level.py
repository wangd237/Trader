"""验证 Gemini 3.x 的 thinking_level 参数传递。

Gemini 3.x 使用字符串参数；Pro、3.8+ 和 -latest 别名
不接受 "minimal"，应映射为 "low"；
3.8 之前的明确版本 Flash 模型仍可使用 "minimal"。"""

from unittest import mock

import pytest

from tradingagents.llm_clients.google_client import GoogleClient


def _captured_kwargs(model, **kwargs):
    captured = {}
    with mock.patch.object(
        __import__("tradingagents.llm_clients.google_client", fromlist=["x"]),
        "NormalizedChatGoogleGenerativeAI",
        lambda **kw: captured.setdefault("kw", kw),
    ):
        GoogleClient(model, api_key="x", **kwargs).get_llm()
    return captured["kw"]


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
def test_flash_passes_thinking_level_through(level):
    kw = _captured_kwargs("gemini-3.5-flash", thinking_level=level)
    assert kw["thinking_level"] == level
    assert "thinking_budget" not in kw  # the 2.5-era param is gone


def test_pro_remaps_minimal_to_low():
    kw = _captured_kwargs("gemini-3.1-pro-preview", thinking_level="minimal")
    assert kw["thinking_level"] == "low"  # Pro doesn't accept "minimal"


def test_flash_38_remaps_minimal_to_low():
    kw = _captured_kwargs("gemini-3.8-flash", thinking_level="minimal")
    assert kw["thinking_level"] == "low"  # 3.8 Flash 400s on "minimal"


def test_flash_38_keeps_supported_levels():
    kw = _captured_kwargs("gemini-3.8-flash", thinking_level="high")
    assert kw["thinking_level"] == "high"


@pytest.mark.parametrize("alias", ["gemini-flash-latest", "gemini-pro-latest"])
def test_latest_alias_remaps_minimal_to_low(alias):
    # 模型别名会随版本变化；gemini-flash-latest 对 "minimal" 返回 400。
    assert _captured_kwargs(alias, thinking_level="minimal")["thinking_level"] == "low"


def test_pro_keeps_high():
    kw = _captured_kwargs("gemini-3.1-pro-preview", thinking_level="high")
    assert kw["thinking_level"] == "high"


def test_no_thinking_level_is_omitted():
    kw = _captured_kwargs("gemini-3.5-flash")
    assert "thinking_level" not in kw
    assert "thinking_budget" not in kw


@pytest.mark.unit
def test_a_gemini_call_times_out_like_the_other_providers():
    """OpenAI 与 Anthropic SDK 默认 600 秒超时，
    Gemini 若未指定超时可能一直等待，导致调用挂起（#1417）。"""
    assert _captured_kwargs("gemini-3.5-flash")["timeout"] == 600
    assert _captured_kwargs("gemini-3.5-flash", timeout=30)["timeout"] == 30
