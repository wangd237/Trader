"""测试 Anthropic effort 参数是否按模型能力正确启用（#831）。"""

import pytest

from tradingagents.llm_clients import anthropic_client as mod


def _capture_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedChatAnthropic",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


@pytest.mark.unit
class TestEffortGate:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5", "claude-haiku-5-0", "claude-haiku-4-7-preview",
            # Sonnet 4.5 及更早版本不支持 effort，会返回 HTTP 400；
            # Sonnet 4.6 及以上版本才支持。
            "claude-sonnet-4-5", "claude-sonnet-4-0",
        ],
    )
    def test_unsupported_models_do_not_receive_effort(self, monkeypatch, model):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model=model, effort="medium", api_key="x").get_llm()
        assert "effort" not in captured["kwargs"]

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7",
            "claude-sonnet-4-6",
        ],
    )
    def test_current_opus_and_sonnet_receive_effort(self, monkeypatch, model):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model=model, effort="high", api_key="x").get_llm()
        assert captured["kwargs"]["effort"] == "high"

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5-0", "claude-opus-4-8", "claude-sonnet-5-0"],
    )
    def test_future_opus_sonnet_inherit_effort_via_pattern(self, monkeypatch, model):
        """验证对未来 Opus、Sonnet 版本的兼容性，无须修改代码。"""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model=model, effort="low", api_key="x").get_llm()
        assert captured["kwargs"]["effort"] == "low"

    @pytest.mark.parametrize(
        "model",
        # Claude 5 家族采用单数字版本 ID，且都支持 effort。
        ["claude-sonnet-5", "claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-opus-5-5",
         "claude-fable-5-1"],
    )
    def test_claude_5_family_receives_effort(self, monkeypatch, model):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(model=model, effort="high", api_key="x").get_llm()
        assert captured["kwargs"]["effort"] == "high"

    def test_mythos_preview_receives_effort(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-mythos-preview", effort="medium", api_key="x"
        ).get_llm()
        assert captured["kwargs"]["effort"] == "medium"

    def test_unknown_anthropic_model_does_not_receive_effort(self, monkeypatch):
        """默认采用保守策略：未知模型不传 effort，避免 HTTP 400。"""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-experimental-x", effort="medium", api_key="x"
        ).get_llm()
        assert "effort" not in captured["kwargs"]

    def test_other_kwargs_still_forwarded_when_effort_skipped(self, monkeypatch):
        """移除 effort 参数时不应影响其他透传参数。"""
        captured = _capture_kwargs(monkeypatch)
        mod.AnthropicClient(
            model="claude-haiku-4-5",
            effort="medium",
            api_key="placeholder",
            max_tokens=1024,
            timeout=30,
        ).get_llm()
        assert captured["kwargs"]["api_key"] == "placeholder"
        assert captured["kwargs"]["max_tokens"] == 1024
        assert captured["kwargs"]["timeout"] == 30
        assert "effort" not in captured["kwargs"]
