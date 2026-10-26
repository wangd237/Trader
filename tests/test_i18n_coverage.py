"""所有输出会进入报告的 Agent 都必须遵守配置的语言（#740/#801）。

非英语运行应完整本地化报告，不能混用英文。
此前部分 Agent 漏传语言指令（在 6b384f7 中修复），
测试防止后续重构再次出现该问题。"""
from pathlib import Path

import pytest

from tradingagents.agents.context import get_language_instruction

_AGENTS_DIR = Path(__file__).resolve().parents[1] / "tradingagents" / "agents"

# 所有会向最终保存报告贡献内容的节点都应覆盖。
# 新增报告 Agent 时，也需要调用 get_language_instruction()。
REPORT_AGENTS = [
    "analysts/market_analyst.py",
    "analysts/news_analyst.py",
    "analysts/fundamentals_analyst.py",
    "analysts/sentiment_analyst.py",
    "researchers/bull_researcher.py",
    "researchers/bear_researcher.py",
    "managers/research_manager.py",
    "managers/portfolio_manager.py",
    "risk_mgmt/aggressive_debator.py",
    "risk_mgmt/conservative_debator.py",
    "risk_mgmt/neutral_debator.py",
    "trader/trader.py",
]


@pytest.mark.unit
class TestLanguageInstruction:
    def test_english_adds_no_tokens(self, monkeypatch):
        from tradingagents.dataflows.config import set_config
        set_config({"output_language": "English"})
        assert get_language_instruction() == ""

    def test_non_english_emits_directive(self):
        from tradingagents.dataflows.config import set_config
        set_config({"output_language": "中文"})
        out = get_language_instruction()
        assert "中文" in out
        assert "entire response" in out

    def test_the_labelled_lines_stay_as_the_format_gives_them(self):
        # 评级依靠固定的 "**Rating**:" 标签解析；如果标签被翻译，
        # 后续只能从自然语言正文猜测，带否定的句子（如“不建议卖出 (Sell)”）
        # 可能使 Buy 被误读成 Sell（#1435）。
        from tradingagents.dataflows.config import set_config
        set_config({"output_language": "中文"})
        out = get_language_instruction()
        assert "**Rating**:" in out and "FINAL TRANSACTION PROPOSAL" in out


@pytest.mark.unit
@pytest.mark.parametrize("rel", REPORT_AGENTS)
def test_report_agent_applies_language_instruction(rel):
    path = _AGENTS_DIR / rel
    assert path.exists(), f"missing agent module: {rel}"
    src = path.read_text(encoding="utf-8")
    assert "get_language_instruction()" in src, (
        f"{rel} does not apply get_language_instruction(); its output would "
        f"ignore the configured output_language (#740/#801)."
    )
