"""测试环境变量驱动的 CLI 行为（#897、#873）。"""

import os
import unittest
from unittest import mock

import pytest

import cli.selections as cli_selections


@pytest.mark.unit
class TestProviderDefaultUrl(unittest.TestCase):
    def test_known_providers_resolve(self):
        from cli.prompts import provider_default_url
        self.assertEqual(provider_default_url("openai"), "https://api.openai.com/v1")
        self.assertEqual(provider_default_url("DeepSeek"), "https://api.deepseek.com")
        self.assertIsNone(provider_default_url("google"))  # uses SDK default

    def test_unknown_provider_returns_none(self):
        from cli.prompts import provider_default_url
        self.assertIsNone(provider_default_url("not-a-provider"))

    def test_ollama_honors_base_url_env(self):
        from cli.prompts import provider_default_url
        with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://host:1234/v1"}):
            self.assertEqual(provider_default_url("ollama"), "http://host:1234/v1")


@pytest.mark.unit
class TestCliSkipsPromptsFromEnv(unittest.TestCase):
    def test_env_config_skips_llm_prompts(self):

        env = {
            "TRADINGAGENTS_LLM_PROVIDER": "openai",
            "TRADINGAGENTS_DEEP_THINK_LLM": "kimi-k2.5",
            "TRADINGAGENTS_QUICK_THINK_LLM": "deepseek-v4-pro",
            "TRADINGAGENTS_LLM_BACKEND_URL": "https://opencode.ai/zen/go/v1",
            "TRADINGAGENTS_OUTPUT_LANGUAGE": "Japanese",
        }
        fake_cfg = dict(cli_selections.DEFAULT_CONFIG)
        fake_cfg.update({
            "llm_provider": "openai",
            "backend_url": "https://opencode.ai/zen/go/v1",
            "quick_think_llm": "deepseek-v4-pro",
            "deep_think_llm": "kimi-k2.5",
            "output_language": "Japanese",
        })

        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(cli_selections, "DEFAULT_CONFIG", fake_cfg), \
             mock.patch.object(cli_selections.updates, "enabled", return_value=False), \
             mock.patch.object(cli_selections, "get_ticker", return_value="AAPL"), \
             mock.patch.object(cli_selections, "get_analysis_date", return_value="2026-05-29"), \
             mock.patch.object(cli_selections, "select_analysts", return_value=[]), \
             mock.patch.object(cli_selections, "select_research_depth", return_value=1), \
             mock.patch.object(cli_selections, "ensure_api_key") as ensure_key, \
             mock.patch.object(cli_selections, "select_llm_provider") as prompt_provider, \
             mock.patch.object(cli_selections, "ask_output_language") as prompt_lang, \
             mock.patch.object(cli_selections, "select_shallow_thinking_agent") as prompt_quick, \
             mock.patch.object(cli_selections, "select_deep_thinking_agent") as prompt_deep:
            sel = cli_selections.get_user_selections()

        # 不应显示任何 LLM 选择问题。
        prompt_provider.assert_not_called()
        prompt_lang.assert_not_called()
        prompt_quick.assert_not_called()
        prompt_deep.assert_not_called()
        # 即使提供商由环境变量指定，也必须校验 API Key。
        ensure_key.assert_called_once()

        # 返回的选择结果应正确包含环境变量的值。
        self.assertEqual(sel["llm_provider"], "openai")
        self.assertEqual(sel["backend_url"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(sel["quick_think_llm"], "deepseek-v4-pro")
        self.assertEqual(sel["deep_think_llm"], "kimi-k2.5")
        self.assertEqual(sel["output_language"], "Japanese")


@pytest.mark.unit
class TestResearchDepthSkippedFromEnv(unittest.TestCase):
    def test_both_round_envs_skip_depth_prompt(self):

        env = {
            "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "2",
            "TRADINGAGENTS_MAX_RISK_ROUNDS": "4",
        }
        fake_cfg = dict(cli_selections.DEFAULT_CONFIG)
        fake_cfg.update({"max_debate_rounds": 2, "max_risk_discuss_rounds": 4})

        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(cli_selections, "DEFAULT_CONFIG", fake_cfg), \
             mock.patch.object(cli_selections.updates, "enabled", return_value=False), \
             mock.patch.object(cli_selections, "get_ticker", return_value="AAPL"), \
             mock.patch.object(cli_selections, "get_analysis_date", return_value="2026-05-29"), \
             mock.patch.object(cli_selections, "select_analysts", return_value=[]), \
             mock.patch.object(cli_selections, "select_research_depth") as prompt_depth, \
             mock.patch.object(cli_selections, "ensure_api_key"), \
             mock.patch.object(cli_selections, "select_llm_provider", return_value=("openai", None)), \
             mock.patch.object(cli_selections, "ask_output_language", return_value="English"), \
             mock.patch.object(cli_selections, "select_shallow_thinking_agent", return_value="gpt-5.4-mini"), \
             mock.patch.object(cli_selections, "select_deep_thinking_agent", return_value="gpt-5.5"), \
             mock.patch.object(cli_selections, "ask_openai_reasoning_effort", return_value=None):
            sel = cli_selections.get_user_selections()

        # 跳过研究深度提问，直接使用环境变量配置。
        prompt_depth.assert_not_called()
        self.assertEqual(sel["research_depth"], 2)


@pytest.mark.unit
class TestReasoningEffortSkippedFromEnv(unittest.TestCase):
    def test_effort_env_skips_step8_prompt(self):

        env = {"TRADINGAGENTS_OPENAI_REASONING_EFFORT": "high"}
        fake_cfg = dict(cli_selections.DEFAULT_CONFIG)
        fake_cfg.update({"openai_reasoning_effort": "high"})

        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(cli_selections, "DEFAULT_CONFIG", fake_cfg), \
             mock.patch.object(cli_selections.updates, "enabled", return_value=False), \
             mock.patch.object(cli_selections, "get_ticker", return_value="AAPL"), \
             mock.patch.object(cli_selections, "get_analysis_date", return_value="2026-05-29"), \
             mock.patch.object(cli_selections, "select_analysts", return_value=[]), \
             mock.patch.object(cli_selections, "select_research_depth", return_value=1), \
             mock.patch.object(cli_selections, "ensure_api_key"), \
             mock.patch.object(cli_selections, "select_llm_provider", return_value=("openai", None)), \
             mock.patch.object(cli_selections, "ask_output_language", return_value="English"), \
             mock.patch.object(cli_selections, "select_shallow_thinking_agent", return_value="gpt-5.4-mini"), \
             mock.patch.object(cli_selections, "select_deep_thinking_agent", return_value="gpt-5.5"), \
             mock.patch.object(cli_selections, "ask_openai_reasoning_effort") as prompt_effort:
            sel = cli_selections.get_user_selections()

        # 跳过推理力度提问，直接使用环境变量配置。
        prompt_effort.assert_not_called()
        self.assertEqual(sel["openai_reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
