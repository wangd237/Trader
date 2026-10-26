"""测试 DeepSeekChatOpenAI 思考模式行为。

验证两点：收到模型的 reasoning_content 时放入 AIMessage.additional_kwargs，
下一轮请求必须原样回传；以及结构化输出通过能力表
为 V4、reasoner 禁用 tool_choice，遵循 DeepSeek 官方工具调用模式。"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel

from tradingagents.llm_clients.openai_client import (
    DeepSeekChatOpenAI,
    NormalizedChatOpenAI,
    _input_to_messages,
)

# ---------------------------------------------------------------------------
# _input_to_messages — the helper that handles list / ChatPromptValue / other
# (Gemini bot review note: non-list inputs must also work)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputToMessages:
    def test_list_input_returned_as_is(self):
        msgs = [HumanMessage(content="hi")]
        assert _input_to_messages(msgs) is msgs

    def test_chat_prompt_value_unwrapped(self):
        msgs = [HumanMessage(content="hi")]
        prompt_value = ChatPromptValue(messages=msgs)
        assert _input_to_messages(prompt_value) == msgs

    def test_string_input_yields_empty_list(self):
        # 普通字符串本身不包含消息对象；
        # LangChain 转换会在 _get_request_payload 之前完成。
        assert _input_to_messages("hello") == []


# ---------------------------------------------------------------------------
# Reasoning content propagation across turns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeepSeekReasoningContent:
    def _client(self):
        os.environ.setdefault("DEEPSEEK_API_KEY", "placeholder")
        return DeepSeekChatOpenAI(
            model="deepseek-v4-flash",
            api_key="placeholder",
            base_url="https://api.deepseek.com",
        )

    def test_capture_on_receive(self):
        """响应中包含 reasoning_content 时，必须保存到 AIMessage.additional_kwargs，
        让下轮请求能够回传。"""
        client = self._client()
        result = client._create_chat_result(
            {
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Plan: buy NVDA.",
                            "reasoning_content": "Step 1: trend is up. Step 2: ...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        ai = result.generations[0].message
        assert ai.additional_kwargs["reasoning_content"] == "Step 1: trend is up. Step 2: ..."

    def test_propagate_on_send(self):
        """向模型发送 AIMessage 时，必须将 reasoning_content 注入对应请求消息。"""
        client = self._client()
        prior = AIMessage(
            content="Plan",
            additional_kwargs={"reasoning_content": "weighed bull case"},
        )
        new_user = HumanMessage(content="Refine.")
        payload = client._get_request_payload([prior, new_user])
        # 在请求负载中找到 assistant 消息。
        assistant_dicts = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert assistant_dicts, "assistant message missing from outgoing payload"
        assert assistant_dicts[0]["reasoning_content"] == "weighed bull case"

    def test_propagate_through_chat_prompt_value(self):
        """除列表外，还要支持 ChatPromptValue 输入携带 reasoning_content。"""
        client = self._client()
        prior = AIMessage(
            content="Plan",
            additional_kwargs={"reasoning_content": "weighed bull case"},
        )
        prompt_value = ChatPromptValue(messages=[prior, HumanMessage(content="Refine.")])
        payload = client._get_request_payload(prompt_value)
        assistant_dicts = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert assistant_dicts[0]["reasoning_content"] == "weighed bull case"


# ---------------------------------------------------------------------------
# Capability-driven structured output: tool_choice suppressed for V4 + reasoner
# ---------------------------------------------------------------------------


def _bound_kwargs(runnable):
    """从 with_structured_output 的结果中提取 bind() 参数。"""
    first = runnable.steps[0] if hasattr(runnable, "steps") else runnable
    return getattr(first, "kwargs", {})


@pytest.mark.unit
class TestStructuredOutputCapabilityDispatch:
    """DeepSeek V4 和 reasoner 会拒绝 tool_choice；
    官方工具调用示例只发送 tools=[...]。
    验证能力分发会禁用该参数，但普通 chat 模型仍正常发送。"""

    class _Sample(BaseModel):
        answer: str

    def _client(self, model):
        return DeepSeekChatOpenAI(
            model=model, api_key="placeholder", base_url="https://api.deepseek.com",
        )

    def test_chat_sends_tool_choice(self):
        bound = self._client("deepseek-chat").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is not None

    def test_reasoner_suppresses_tool_choice(self):
        bound = self._client("deepseek-reasoner").with_structured_output(self._Sample)
        # tool_choice 缺失或显式为 None 都意味着不强制工具，
        # LangChain 的 bind_tools 应跳过该参数。
        assert _bound_kwargs(bound).get("tool_choice") in (None, ...) or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_v4_flash_suppresses_tool_choice(self):
        bound = self._client("deepseek-v4-flash").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is None or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_v4_pro_suppresses_tool_choice(self):
        bound = self._client("deepseek-v4-pro").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is None or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_future_v_variant_via_regex(self):
        """向前兼容：未知 deepseek-v\\d-* 模型继承 V4 兼容规则。"""
        bound = self._client("deepseek-v5-hypothetical").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is None or \
            "tool_choice" not in _bound_kwargs(bound)

    def test_schema_is_still_bound_as_tool(self):
        """虽然禁用 tool_choice，仍须将 Schema 作为工具绑定，
        与 DeepSeek 官方工具调用示例一致。"""
        bound = self._client("deepseek-reasoner").with_structured_output(self._Sample)
        kwargs = _bound_kwargs(bound)
        tools = kwargs.get("tools", [])
        assert any(
            t.get("function", {}).get("name") == "_Sample" for t in tools
        ), f"schema not bound as a tool: {tools}"


# ---------------------------------------------------------------------------
# Live API: structured output round-trips against the real DeepSeek backend
# ---------------------------------------------------------------------------


def _has_real_deepseek_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    return bool(key) and key != "placeholder"


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_real_deepseek_key(),
    reason="DEEPSEEK_API_KEY not set (or placeholder); skipping live API call",
)
class TestDeepSeekLiveStructuredOutput:
    """端到端测试：真实 DeepSeek V4 Flash 调用应返回结构化实例。
    验证不传 tool_choice 能避免 #678 的 400 错误，
    并保持 Pydantic 结果解析能力。"""

    class _Pick(BaseModel):
        action: str
        confidence: float

    def test_v4_flash_returns_structured_output(self):
        client = DeepSeekChatOpenAI(
            model="deepseek-v4-flash",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=60,
        )
        bound = client.with_structured_output(self._Pick)
        result = bound.invoke(
            "Pick BUY or SELL or HOLD for a tech stock with strong earnings. "
            "Confidence is a float between 0 and 1."
        )
        assert isinstance(result, self._Pick)
        assert result.action in {"BUY", "SELL", "HOLD"}
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Base class isolation: NormalizedChatOpenAI does NOT have DeepSeek behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseClassIsolation:
    def test_normalized_does_not_propagate_reasoning_content(self):
        """通用 NormalizedChatOpenAI 不应包含 DeepSeek 专属行为；
        只有其子类负责这些兼容处理。"""
        assert not hasattr(NormalizedChatOpenAI, "_get_request_payload") or (
            NormalizedChatOpenAI._get_request_payload
            is NormalizedChatOpenAI.__bases__[0]._get_request_payload
        )
