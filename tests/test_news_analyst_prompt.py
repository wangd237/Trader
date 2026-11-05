"""验证新闻分析师 Prompt 与工具签名保持一致（#1116）。

旧 Prompt 错误地告知模型可调用 ``get_news(query, ...)``，
诱导其编造自由文本 query 参数。
现在标的从运行状态提供，模型只需要传入分析窗口。"""
import inspect

import pytest

import tradingagents.agents.analysts.news_analyst as na
from tradingagents.agents.tools import get_news


@pytest.mark.unit
def test_get_news_takes_the_window_not_a_query():
    arg_names = set(get_news.tool_call_schema.model_json_schema()["properties"])
    assert arg_names == {"start_date", "end_date"}


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    src = inspect.getsource(na)
    assert "get_news(start_date, end_date)" in src
    assert "get_news(query" not in src



@pytest.mark.unit
def test_hk_news_analyst_uses_hkex_not_us_macro_yahoo_or_fred(monkeypatch):
    from langchain_core.messages import AIMessage

    from tradingagents.agents.tools import get_macro_indicators, get_news

    captured = {}

    def fake_turn(prompt, llm, tools, messages):
        captured["tools"] = tools
        captured["prompt"] = "\n".join(m.content for m in prompt.format_messages(messages=messages))
        return AIMessage(content="HKEX report"), "HKEX report"

    monkeypatch.setattr(na, "take_turn", fake_turn)
    node = na.create_news_analyst(object())
    result = node({
        "company_of_interest": "9988.HK",
        "trade_date": "2026-11-04",
        "messages": [],
    })
    assert result["news_report"] == "HKEX report"
    assert captured["tools"] == (get_news,)
    assert get_macro_indicators not in captured["tools"]
    assert "HKEXnews" in captured["prompt"]
    assert "Do not claim access to global media, FRED" in captured["prompt"]
    assert "get_macro_indicators," not in captured["prompt"]
