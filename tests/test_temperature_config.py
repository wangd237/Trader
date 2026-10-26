"""测试可配置的采样温度（#178/#168）。

temperature 是跨提供商参数；设置后必须传给底层聊天客户端，
未设置时则保留提供商自身默认值。"""


import pytest

from tradingagents.llm_clients.factory import create_llm_client


@pytest.mark.unit
class TestTemperatureForwarding:
    @pytest.mark.parametrize(
        "provider,model",
        [
            # 故意使用非推理模型 gpt-4.1 测试 temperature 透传；
            # GPT-5 系列属于推理模型，会正确忽略该参数，
            # 相关逻辑由 test_openai_reasoning_effort 检查。
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-5"),
            ("google", "gemini-3.5-flash"),
            ("deepseek", "deepseek-chat"),
        ],
    )
    def test_temperature_reaches_client_when_set(self, provider, model):
        llm = create_llm_client(
            provider=provider, model=model, temperature=0.0, api_key="placeholder"
        ).get_llm()
        assert llm.temperature == 0.0

    def test_temperature_omitted_leaves_provider_default(self):
        # 未传 temperature 时不得强制设置默认数值。
        llm = create_llm_client(
            provider="openai", model="gpt-4.1", api_key="placeholder"
        ).get_llm()
        # LangChain 默认值为未设置或 None，而不是 0.0。
        assert llm.temperature is None


@pytest.mark.unit
class TestTemperatureEnvOverlay:
    def test_env_sets_temperature(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.setenv("TRADINGAGENTS_TEMPERATURE", "0.2")
        config = dc.build_default_config()
        # 配置项中允许保存环境变量字符串，使用时通过 float() 转换。
        assert config["temperature"] in ("0.2", 0.2)
        assert float(config["temperature"]) == 0.2

    def test_default_temperature_is_none(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.delenv("TRADINGAGENTS_TEMPERATURE", raising=False)
        assert dc.build_default_config()["temperature"] is None


@pytest.mark.unit
class TestProviderKwargsTemperature:
    """build_llm_kwargs 将 temperature 转换为浮点数并透传，未提供则省略。"""

    def _kwargs_for(self, temperature):
        from tradingagents.llm_clients import build_llm_kwargs
        return build_llm_kwargs({"llm_provider": "openai", "temperature": temperature})

    def test_float_string_coerced(self):
        assert self._kwargs_for("0.3")["temperature"] == 0.3

    def test_float_passthrough(self):
        assert self._kwargs_for(0.0)["temperature"] == 0.0

    def test_none_omitted(self):
        assert "temperature" not in self._kwargs_for(None)

    def test_empty_string_omitted(self):
        assert "temperature" not in self._kwargs_for("")
