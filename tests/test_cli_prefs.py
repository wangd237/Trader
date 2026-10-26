"""CLI 记住用户上一次的选择，并在下次启动时提供预选值。

仅预填而不跳过问题，确保用户能看到选项。
环境变量优先级更高，仍可跳过相应交互步骤。
由于模型和提供商会更新，所有保存的选择都必须再次验证。"""

from __future__ import annotations

from unittest import mock

import pytest

import cli.selections as cli_selections
from cli.models import AnalystType
from cli.prefs import load_last_run, sanitize, save_last_run

SAVED = {
    "output_language": "English",
    "analysts": ["market", "fundamentals"],
    "research_depth": 3,
    "llm_provider": "openai",
    "quick_think_llm": "gpt-5.6-mini",
    "deep_think_llm": "gpt-5.6",
}


@pytest.mark.unit
def test_round_trip():
    save_last_run(SAVED)
    assert load_last_run() == SAVED


@pytest.mark.unit
def test_missing_file_is_not_an_error():
    assert load_last_run() == {}


@pytest.mark.unit
def test_a_corrupt_file_degrades_to_no_memory(tmp_path):
    (tmp_path / "cli_prefs.json").write_text("{not json")
    assert load_last_run() == {}


@pytest.mark.unit
def test_a_half_written_file_cannot_be_observed(tmp_path):
    """两个运行同时结束时，偏好配置文件不能出现部分写入。"""
    save_last_run(SAVED)
    save_last_run({**SAVED, "research_depth": 5})
    assert load_last_run()["research_depth"] == 5
    assert list(tmp_path.glob("*.tmp*")) == []


# --- 校验当前可选项 ---------------------------------

@pytest.mark.unit
def test_a_model_that_no_longer_exists_is_dropped():
    # gpt-5.4 仍被配置层接受，但已从交互选择菜单中移除。
    kept = sanitize({**SAVED, "quick_think_llm": "gpt-5.4"}, "stock")
    assert "quick_think_llm" not in kept
    assert kept["deep_think_llm"] == "gpt-5.6"  # the valid sibling survives


@pytest.mark.unit
def test_an_unknown_provider_drops_itself_and_its_models():
    kept = sanitize({**SAVED, "llm_provider": "no-such-provider"}, "stock")
    assert "llm_provider" not in kept
    assert "quick_think_llm" not in kept and "deep_think_llm" not in kept


@pytest.mark.unit
def test_analysts_are_narrowed_to_the_asset_type():
    kept = sanitize(SAVED, "crypto")
    assert AnalystType.FUNDAMENTALS.value not in kept["analysts"]
    assert AnalystType.MARKET.value in kept["analysts"]


@pytest.mark.unit
def test_junk_values_are_dropped_rather_than_offered():
    kept = sanitize({"research_depth": 99, "analysts": ["astrology"], "output_language": 5}, "stock")
    assert kept == {}


@pytest.mark.unit
def test_a_region_specific_provider_survives():
    kept = sanitize({**SAVED, "llm_provider": "qwen-cn", "quick_think_llm": None}, "stock")
    assert kept["llm_provider"] == "qwen-cn"


# --- 流程连接 ------------------------------------------------------------------

def _answer_every_prompt(monkeypatch):
    """真实执行选择流程，每个交互问题由固定答案响应。"""
    import cli.main as m

    monkeypatch.setattr(cli_selections.updates, "enabled", lambda: False)
    monkeypatch.setattr(cli_selections, "get_ticker", lambda: "NVDA")
    monkeypatch.setattr(cli_selections, "get_analysis_date", lambda: "2026-09-01")
    monkeypatch.setattr(cli_selections, "ask_output_language", lambda default=None: "English")
    monkeypatch.setattr(cli_selections, "select_analysts", lambda asset_type, default=None: [AnalystType.MARKET])
    monkeypatch.setattr(cli_selections, "select_research_depth", lambda default=None: 3)
    monkeypatch.setattr(cli_selections, "select_llm_provider", lambda default=None: ("openai", None))
    monkeypatch.setattr(cli_selections, "select_shallow_thinking_agent", lambda p, default=None: "gpt-5.6-mini")
    monkeypatch.setattr(cli_selections, "select_deep_thinking_agent", lambda p, default=None: "gpt-5.6")
    monkeypatch.setattr(cli_selections, "ask_openai_reasoning_effort", lambda: "medium")
    return m


@pytest.mark.unit
def test_selections_are_remembered_after_a_run(monkeypatch):
    """直接伪造 selections 字典会掩盖配置键不匹配，必须测试真实流程。"""
    _answer_every_prompt(monkeypatch)

    cli_selections.get_user_selections()

    remembered = load_last_run()
    assert remembered["analysts"] == ["market"]
    assert remembered["quick_think_llm"] == "gpt-5.6-mini"
    assert remembered["deep_think_llm"] == "gpt-5.6"
    assert remembered["llm_provider"] == "openai"
    assert "ticker" not in remembered  # changes every run; never remembered
    assert "analysis_date" not in remembered  # a stale date must not be offered


@pytest.mark.unit
def test_a_custom_language_is_remembered_without_breaking_the_next_run():
    """自由输入值可能不在菜单选项内，questionary 会拒绝将它预设为默认值，
    直接传入可能让 CLI 启动崩溃。"""
    from cli.prompts import ask_output_language

    save_last_run({"output_language": "Turkish"})
    with mock.patch("cli.prompts.questionary.select") as select:
        select.return_value.ask.return_value = "English"
        ask_output_language(load_last_run()["output_language"])
    assert select.call_args.kwargs["default"] is None


@pytest.mark.unit
def test_a_remembered_endpoint_is_offered_back(monkeypatch):
    """本地或自定义端点的 URL 虽曾保存并校验，
    但未被用于下次预填，导致用户反复输入。"""

    save_last_run({"llm_provider": "openai_compatible", "backend_url": "http://localhost:1234/v1"})
    offered = {}
    monkeypatch.setattr(cli_selections, "select_llm_provider", lambda default=None: ("openai_compatible", None))
    monkeypatch.setattr(cli_selections, "prompt_openai_compatible_url",
                        lambda default=None: offered.setdefault("default", default) or "http://x/v1")
    monkeypatch.setattr(cli_selections.updates, "enabled", lambda: False)
    monkeypatch.setattr(cli_selections, "get_ticker", lambda: "NVDA")
    monkeypatch.setattr(cli_selections, "get_analysis_date", lambda: "2026-09-01")
    monkeypatch.setattr(cli_selections, "ask_output_language", lambda default=None: "English")
    monkeypatch.setattr(cli_selections, "select_analysts", lambda asset_type, default=None: [AnalystType.MARKET])
    monkeypatch.setattr(cli_selections, "select_research_depth", lambda default=None: 1)
    monkeypatch.setattr(cli_selections, "select_shallow_thinking_agent", lambda p, default=None: "local-model")
    monkeypatch.setattr(cli_selections, "select_deep_thinking_agent", lambda p, default=None: "local-model")

    cli_selections.get_user_selections()

    assert offered["default"] == "http://localhost:1234/v1"
