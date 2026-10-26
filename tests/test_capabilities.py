"""LLM 模型能力表的单元测试。"""

from dataclasses import FrozenInstanceError

import pytest

from tradingagents.llm_clients.capabilities import (
    get_capabilities,
)


@pytest.mark.unit
class TestExactIdMatches:
    def test_deepseek_chat_supports_tool_choice(self):
        caps = get_capabilities("deepseek-chat")
        assert caps.supports_tool_choice is True

    def test_deepseek_reasoner_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-reasoner")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_deepseek_v4_flash_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-v4-flash")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_deepseek_flash_alias_rejects_tool_choice(self):
        """模型选择器使用 ``deepseek-flash`` 表示 V4.1 Flash。
        此前它错误落到 _DEFAULT（启用 tool_choice），
        使结构化输出收到 HTTP 400 并退化为文本重试。"""
        caps = get_capabilities("deepseek-flash")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_deepseek_v4_pro_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-v4-pro")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True


@pytest.mark.unit
class TestPatternMatches:
    """验证前向兼容正则能识别未知 DeepSeek 与 MiniMax 变体。"""

    def test_future_deepseek_v5_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-v5-flash")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_future_deepseek_v9_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-v9-anything")
        assert caps.supports_tool_choice is False

    def test_reasoner_variant_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-reasoner-pro")
        assert caps.supports_tool_choice is False

    def test_future_flash_variant_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-flash-lite")
        assert caps.supports_tool_choice is False

    def test_minimax_m3_inherits_thinking_quirks(self):
        caps = get_capabilities("MiniMax-M3")
        assert caps.supports_tool_choice is False

    def test_future_minimax_m4_highspeed_inherits_thinking_quirks(self):
        caps = get_capabilities("MiniMax-M4-highspeed")
        assert caps.supports_tool_choice is False


@pytest.mark.unit
class TestMinimaxExactMatches:
    """MiniMax M2.x 不接受 LangChain 使用的函数描述字典 tool_choice；
    官方接口仅允许 none 或 auto。"""

    def test_m2_7_rejects_tool_choice(self):
        caps = get_capabilities("MiniMax-M2.7")
        assert caps.supports_tool_choice is False
        assert caps.supports_json_mode is False  # only MiniMax-Text-01 supports json_object

    def test_m2_7_highspeed_rejects_tool_choice(self):
        assert get_capabilities("MiniMax-M2.7-highspeed").supports_tool_choice is False

    def test_m2_1_rejects_tool_choice(self):
        assert get_capabilities("MiniMax-M2.1").supports_tool_choice is False

    def test_m2_base_rejects_tool_choice(self):
        assert get_capabilities("MiniMax-M2").supports_tool_choice is False

    def test_m2_x_requires_reasoning_split(self):
        # M2.x 推理模型要求 reasoning_split=True，
        # 使 <think> 块存入 reasoning_details 而非普通 content（#826）。
        for model in ("MiniMax-M2.7", "MiniMax-M2.5-highspeed", "MiniMax-M2"):
            assert get_capabilities(model).requires_reasoning_split is True

    def test_future_m3_inherits_reasoning_split(self):
        assert get_capabilities("MiniMax-M3-highspeed").requires_reasoning_split is True

    def test_non_reasoning_minimax_does_not_get_reasoning_split(self):
        # Coding Plan、MiniMax-Text-01 和非 M2 模型不支持 reasoning_split，
        # OpenAI SDK 的严格参数校验会拒绝该字段；默认必须关闭（#826）。
        for model in ("minimax-text-01", "MiniMax-Coding-Plan", "abab6.5-chat"):
            assert get_capabilities(model).requires_reasoning_split is False


@pytest.mark.unit
class TestDefault:
    """未知或非 DeepSeek 模型采用宽松默认能力。"""

    def test_gpt_default(self):
        caps = get_capabilities("gpt-4.1")
        assert caps.supports_tool_choice is True
        assert caps.preferred_structured_method == "function_calling"

    def test_grok_default(self):
        caps = get_capabilities("grok-4-0709")
        assert caps.supports_tool_choice is True

    def test_unknown_model_default(self):
        caps = get_capabilities("totally-made-up-model-id")
        assert caps.supports_tool_choice is True

    def test_exact_match_precedes_pattern(self):
        """deepseek-chat 不能匹配版本号正则 v\d。"""
        caps = get_capabilities("deepseek-chat")
        assert caps.supports_tool_choice is True


@pytest.mark.unit
class TestOpenRouterDeepSeekNamespace:
    """OpenRouter 使用 ``deepseek/<id>`` 为 DeepSeek 模型添加命名空间；
    剥离该前缀后应应用原生提供商相同的兼容规则（#1199）。"""

    def test_prefixed_v4_flash_suppresses_tool_choice(self):
        # 此前错误使用 _DEFAULT 并启用 tool_choice，造成对象形式调用失败。
        assert get_capabilities("deepseek/deepseek-v4-flash").supports_tool_choice is False

    def test_prefixed_reasoner_suppresses_tool_choice(self):
        assert get_capabilities("deepseek/deepseek-reasoner").supports_tool_choice is False

    def test_prefixed_flash_alias_suppresses_tool_choice(self):
        assert get_capabilities("deepseek/deepseek-flash").supports_tool_choice is False

    def test_prefixed_chat_selects_deepseek_chat_not_default(self):
        # 必须匹配 _DEEPSEEK_CHAT 而非 _DEFAULT：
        # 前者设置 supports_json_schema=False，虽然两者都允许 tool_choice。
        caps = get_capabilities("deepseek/deepseek-chat")
        assert caps.supports_tool_choice is True
        assert caps.supports_json_schema is False  # _DEEPSEEK_CHAT, not _DEFAULT

    def test_only_official_namespace_is_stripped(self):
        # 其他发布者即使模型名包含 deepseek 版本模式，也应使用 _DEFAULT；
        # 只能剥离官方的 deepseek/ 前缀，不能任意剥离其他命名空间。
        caps = get_capabilities("tngtech/deepseek-v4-flash")
        assert caps.supports_tool_choice is True         # not thinking
        assert caps.supports_json_schema is True          # _DEFAULT

    def test_native_ids_unchanged(self):
        assert get_capabilities("deepseek-v4-flash").supports_tool_choice is False
        assert get_capabilities("deepseek-chat").supports_tool_choice is True


@pytest.mark.unit
def test_capabilities_dataclass_is_frozen():
    """模型能力配置不可变，因此可以安全地复用同一对象。"""
    caps = get_capabilities("deepseek-chat")
    with pytest.raises(FrozenInstanceError):
        caps.supports_tool_choice = False  # type: ignore[misc]
