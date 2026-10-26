"""共用路由器的 path_map 完整性测试（#1088）。

风控讨论路由器可返回三个风险节点，多空辩论路由器可返回两个研究节点；
早先各边的映射表并未覆盖完整返回集合。
现在分别共用 RISK_ANALYSIS_PATH_MAP 和 DEBATE_PATH_MAP，
防止发言者标签变化时出现缺失路径，导致 LangGraph 执行中断。"""
import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import DEBATE_PATH_MAP, RISK_ANALYSIS_PATH_MAP


def _state(latest_speaker, count=0):
    return {"risk_debate_state": {"latest_speaker": latest_speaker, "count": count}}


def _debate_state(current_response, count=0):
    return {"investment_debate_state": {"current_response": current_response, "count": count}}


@pytest.mark.unit
@pytest.mark.parametrize("latest_speaker", [
    "Aggressive", "Aggressive Analyst",
    "Conservative", "Conservative Analyst",
    "Neutral", "Neutral Analyst",
    "",                          # drift: empty label
    "Aggressive Risk Analyst",   # drift: node renamed
    "Agresivo",                  # drift: i18n / translated label
])
def test_router_return_always_routable(latest_speaker):
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    target = logic.should_continue_risk_analysis(_state(latest_speaker))
    assert target in RISK_ANALYSIS_PATH_MAP


@pytest.mark.unit
def test_router_terminates_at_round_limit():
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    # 当 count 达到 3 * rounds 时结束辩论并路由到 Portfolio Manager。
    assert logic.should_continue_risk_analysis(_state("Neutral", count=3)) == "Portfolio Manager"


@pytest.mark.unit
def test_path_map_covers_full_router_range():
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    returns = {
        logic.should_continue_risk_analysis(_state(s, c))
        for s in ("Aggressive", "Conservative", "Neutral", "drift")
        for c in (0, 99)
    }
    # 路由器可能产生的每个返回值都必须在共用映射中有对应键。
    assert returns <= set(RISK_ANALYSIS_PATH_MAP)
    # 并且最终终止目标必须可达。
    assert "Portfolio Manager" in returns


@pytest.mark.unit
@pytest.mark.parametrize("current_response", [
    "Bull", "Bull Researcher", "Bear", "Bear Researcher",
    "",                       # drift: empty label
    "Optimista",              # drift: i18n / translated label
])
def test_debate_router_return_always_routable(current_response):
    logic = ConditionalLogic(max_debate_rounds=1)
    target = logic.should_continue_debate(_debate_state(current_response))
    assert target in DEBATE_PATH_MAP


@pytest.mark.unit
def test_debate_path_map_covers_full_router_range():
    logic = ConditionalLogic(max_debate_rounds=1)
    returns = {
        logic.should_continue_debate(_debate_state(s, c))
        for s in ("Bull", "Bear", "drift")
        for c in (0, 99)
    }
    assert returns <= set(DEBATE_PATH_MAP)
    assert "Research Manager" in returns  # terminal reachable
