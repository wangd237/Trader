"""测试 Agent 最终实际收到的提示词。

审计发现过三个问题：有一位分析师的指令被误传为 Python 元组；
所有分析师被要求输出无人使用的交易决策；
未生成的报告被展示成空章节，诱导下游 Agent 凭空补全。"""

from __future__ import annotations

import importlib

import pytest

ANALYSTS = ["market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst"]


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_brief_is_text_not_a_python_object(name):
    """尾部逗号把分析师指令变成了元组，模型收到的是带括号和引号的 repr，
    而非预期的纯文本指令。"""
    import ast
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    tree = ast.parse(inspect.getsource(mod))
    briefs = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and getattr(node.targets[0], "id", "") == "system_message"]
    assert briefs, f"{name} has no system_message"
    for brief in briefs:
        assert not isinstance(brief, ast.Tuple), "the brief is a tuple, not text"


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_is_not_asked_for_a_trade_call_nothing_reads(name):
    """停止信号无人消费；要求分析师给出该信号，
    会让报告先输出方向判断，并被下游错误当作事实依据。"""
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    assert "FINAL TRANSACTION PROPOSAL" not in inspect.getsource(mod)


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", [
    ("tradingagents.agents.researchers.bull_researcher", "create_bull_researcher"),
    ("tradingagents.agents.researchers.bear_researcher", "create_bear_researcher"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
])
def test_a_report_that_was_never_produced_says_so(module, factory):
    """只选择市场分析师时，其余三个报告为空；
    若仍显示空章节，下游模型可能自行编造缺失内容。"""
    from langchain_core.messages import AIMessage

    mod = importlib.import_module(module)
    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt if isinstance(prompt, str) else str(prompt))
            return AIMessage("argument")

        def with_structured_output(self, *a, **k):
            raise NotImplementedError

    state = {
        "company_of_interest": "NVDA", "trade_date": "2026-08-14", "asset_type": "stock",
        "instrument_context": "", "portfolio_context": "", "past_context": "",
        "market_report": "RSI 61, price 178.", "sentiment_report": "", "news_report": "",
        "fundamentals_report": "", "investment_plan": "P", "trader_investment_plan": "T",
        "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "",
                                    "current_response": "", "count": 0},
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": ""},
    }
    getattr(mod, factory)(_LLM())(state)

    prompt = " ".join(seen)
    assert "not part of this run" in prompt or "not available" in prompt, prompt[:400]
    assert "RSI 61" in prompt  # the report that does exist is still passed through
