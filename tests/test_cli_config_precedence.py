"""测试 CLI 配置优先级（#976、#977）。"""

from unittest import mock

import pytest

import cli.main as m
import cli.run as cli_run

# 构造与 get_user_selections() 返回值相同结构的最小选择字典。
SELECTIONS = {
    "research_depth": 5,
    "quick_think_llm": "gpt-5.4-mini",
    "deep_think_llm": "gpt-5.5",
    "backend_url": None,
    "llm_provider": "openai",
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "output_language": "English",
}


def test_research_depth_sets_both_rounds_without_env(monkeypatch):
    for var in ("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "TRADINGAGENTS_MAX_RISK_ROUNDS"):
        monkeypatch.delenv(var, raising=False)
    cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 5
    assert cfg["max_risk_discuss_rounds"] == 5


def test_env_round_counts_win_over_selection(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "2")
    monkeypatch.setenv("TRADINGAGENTS_MAX_RISK_ROUNDS", "4")
    # DEFAULT_CONFIG 在模块导入时已应用环境变量；测试中模拟此行为。
    patched = dict(cli_run.DEFAULT_CONFIG, max_debate_rounds=2, max_risk_discuss_rounds=4)
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 2  # env value, not research_depth=5
    assert cfg["max_risk_discuss_rounds"] == 4


def test_partial_env_only_overrides_that_count(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "2")
    monkeypatch.delenv("TRADINGAGENTS_MAX_RISK_ROUNDS", raising=False)
    patched = dict(cli_run.DEFAULT_CONFIG, max_debate_rounds=2)
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["max_debate_rounds"] == 2  # env wins
    assert cfg["max_risk_discuss_rounds"] == 5  # falls through to research_depth


def test_checkpoint_none_preserves_env_default():
    patched = dict(cli_run.DEFAULT_CONFIG, checkpoint_enabled=True)  # e.g. env-enabled
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=None)
    assert cfg["checkpoint_enabled"] is True  # not clobbered back to False


@pytest.mark.parametrize("flag", [True, False])
def test_checkpoint_flag_overrides_env(flag):
    patched = dict(cli_run.DEFAULT_CONFIG, checkpoint_enabled=not flag)
    with mock.patch.object(cli_run, "DEFAULT_CONFIG", patched):
        cfg = cli_run._build_run_config(SELECTIONS, checkpoint=flag)
    assert cfg["checkpoint_enabled"] is flag


@pytest.mark.unit
def test_glm_resolves_to_the_endpoint_its_key_belongs_to():
    """提供商菜单、客户端注册表及密钥映射必须一致。
    避免某提供商能被选择却无法创建客户端或读取密钥。"""
    from cli.prompts import resolve_backend_url
    from tradingagents.llm_clients.api_key_env import get_api_key_env
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    assert resolve_backend_url("glm", None, None) == OPENAI_COMPATIBLE_PROVIDERS["glm"].base_url
    assert get_api_key_env("glm") == "ZHIPU_API_KEY"
    assert "z.ai" in OPENAI_COMPATIBLE_PROVIDERS["glm"].base_url
    assert "bigmodel.cn" in OPENAI_COMPATIBLE_PROVIDERS["glm-cn"].base_url


@pytest.mark.unit
def test_a_half_set_round_count_says_which_value_won(capsys, monkeypatch):
    """两类轮数只有一个由环境变量配置时，仍应询问研究深度，
    但环境变量指定的那个轮数必须保持优先级。"""

    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1")
    monkeypatch.delenv("TRADINGAGENTS_MAX_RISK_ROUNDS", raising=False)
    printed = []
    monkeypatch.setattr(m.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else ""))

    config = cli_run._build_run_config({
        "ticker": "NVDA", "analysis_date": "2026-09-01", "asset_type": "stock",
        "analysts": [], "research_depth": 5, "llm_provider": "openai",
        "quick_think_llm": "gpt-5.6-luna", "deep_think_llm": "gpt-5.6",
        "backend_url": None, "output_language": "English",
    }, None)

    assert config["max_risk_discuss_rounds"] == 5
    assert any("TRADINGAGENTS_MAX_DEBATE_ROUNDS" in line for line in printed), printed


@pytest.mark.unit
def test_a_tier_s_thinking_setting_survives_a_main_provider_that_has_none(monkeypatch):
    """运行使用 Google 主提供商、OpenAI 子层级时（#1440），
    层级配置必须保留并正确传递，不能被主提供商覆盖。"""
    monkeypatch.setitem(cli_run.DEFAULT_CONFIG, "openai_reasoning_effort", "high")
    monkeypatch.setitem(cli_run.DEFAULT_CONFIG, "anthropic_effort", "low")
    cfg = cli_run._build_run_config({**SELECTIONS, "llm_provider": "google", "google_thinking_level": "high"},
                                    checkpoint=None)
    assert (cfg["openai_reasoning_effort"], cfg["anthropic_effort"], cfg["google_thinking_level"]) == \
        ("high", "low", "high")
