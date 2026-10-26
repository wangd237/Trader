"""测试 MinimaxChatOpenAI 特殊兼容行为。

M2.x 思考模型必须通过 extra_body 发送 reasoning_split=True，
让 <think> 内容分离到 reasoning_details，
避免污染正常回复的 message.content。"""

import os

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tradingagents.llm_clients.openai_client import MinimaxChatOpenAI


def _client(model: str = "MiniMax-M2.7"):
    os.environ.setdefault("MINIMAX_API_KEY", "placeholder")
    return MinimaxChatOpenAI(
        model=model,
        api_key="placeholder",
        base_url="https://api.minimax.io/v1",
    )


@pytest.mark.unit
class TestMinimaxReasoningSplit:
    def test_reasoning_split_sent_via_extra_body_not_top_level(self):
        # 必须通过 extra_body 而非顶层参数发送；OpenAI SDK 会验证顶层字段，
        # 拒绝 reasoning_split 等未知参数（#826）。
        payload = _client()._get_request_payload([HumanMessage(content="hi")])
        assert payload.get("extra_body", {}).get("reasoning_split") is True
        assert "reasoning_split" not in payload  # never top-level

    def test_non_reasoning_minimax_does_not_inject_reasoning_split(self):
        """Coding Plan、MiniMax-Text-01 和非 M2 前缀的模型
        不能发送 reasoning_split（顶层和 extra_body 均不可，#826）。"""
        for model in ("minimax-text-01", "MiniMax-Coding-Plan"):
            payload = _client(model)._get_request_payload(
                [HumanMessage(content="hi")]
            )
            assert "reasoning_split" not in payload
            assert "reasoning_split" not in payload.get("extra_body", {})


@pytest.mark.unit
class TestMinimaxStructuredOutputDispatch:
    """M2.x 模型使用能力表处理兼容性：虽然禁止 tool_choice，
    但仍应将结构化 Schema 绑定为工具。"""

    class _Pick(BaseModel):
        action: str

    def _bound_kwargs(self, runnable):
        first = runnable.steps[0] if hasattr(runnable, "steps") else runnable
        return getattr(first, "kwargs", {})

    def test_m2_7_suppresses_tool_choice(self):
        bound = _client("MiniMax-M2.7").with_structured_output(self._Pick)
        kwargs = self._bound_kwargs(bound)
        assert kwargs.get("tool_choice") is None or "tool_choice" not in kwargs

    def test_m2_7_highspeed_suppresses_tool_choice(self):
        bound = _client("MiniMax-M2.7-highspeed").with_structured_output(self._Pick)
        kwargs = self._bound_kwargs(bound)
        assert kwargs.get("tool_choice") is None or "tool_choice" not in kwargs

    def test_schema_still_bound_as_tool(self):
        bound = _client("MiniMax-M2.7").with_structured_output(self._Pick)
        tools = self._bound_kwargs(bound).get("tools", [])
        assert any(
            t.get("function", {}).get("name") == "_Pick" for t in tools
        ), f"schema not bound: {tools}"
