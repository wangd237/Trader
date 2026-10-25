"""Trader unified config must skip provider/key/model prompts in the CLI."""

import pytest

from cli import selections
from tradingagents.default_config import build_default_config

FLAGS = {
    "ticker": "NVDA", "date": "2026-09-23", "analysts": "market,news",
    "save": True, "show": False,
}


@pytest.mark.unit
def test_unified_env_runs_cli_without_provider_prompts(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setattr(selections, "DEFAULT_CONFIG", build_default_config())
    assert selections.unattended_gaps(FLAGS) == []

    def prohibited(*args, **kwargs):
        raise AssertionError("No model/provider/API key prompt should occur")

    for fn in (
        "select_llm_provider", "select_shallow_thinking_agent",
        "select_deep_thinking_agent", "ensure_api_key",
        "ask_output_language", "select_research_depth",
    ):
        monkeypatch.setattr(selections, fn, prohibited)

    result = selections._prompt_selections({}, FLAGS)
    assert result["llm_provider"] == "deepseek"
    assert result["backend_url"] == "https://api.deepseek.com/v1"
    assert result["quick_think_llm"] == result["deep_think_llm"] == "deepseek-chat"
    assert result["output_language"] == "English"
    assert result["research_depth"] == 1


@pytest.mark.unit
def test_env_example_contains_only_three_llm_fields():
    from pathlib import Path

    env_file = Path(__file__).resolve().parents[1] / ".env.example"
    lines = env_file.read_text(encoding="utf-8").splitlines()
    fields = [line.split("=", 1)[0] for line in lines if line.strip() and not line.startswith("#")]
    assert fields == ["LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"]
