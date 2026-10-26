"""反思模块：将已结算决策的真实结果转化为后续运行可使用的简短经验。"""

from typing import Any


class Reflector:
    """负责对交易决策进行反思。"""

    def __init__(self, quick_thinking_llm: Any):
        """通过 LLM 初始化反思器。"""
        self.quick_thinking_llm = quick_thinking_llm

    def _system_prompt(self, holding_days: int) -> str:
        """为已结算的记忆日志条目构造 reflect_on_final_decision 的简洁提示词。

        生成 2～4 句简短文字，便于注入后续 Agent Prompt，
        而不会显著增大上下文。提示词明确观察窗口的长度，
        因为一周表现不能证伪一个按数月设计的策略；
        若忽略观察期差异，后续运行会将尚不充分的证据误读成确定失败。"""
        return (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            f"The outcome covers {holding_days} trading days from the first close after the analysis date, "
            "which may be shorter than the horizon the decision was written for.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
            "Cover in order:\n"
            f"1. What the {holding_days}-day alpha shows about the directional call (cite the figure), "
            "and say so plainly if the window is too short to judge the thesis.\n"
            "2. Which part of the investment thesis this window supports or undercuts.\n"
            "3. One concrete lesson to apply to the next similar analysis.\n\n"
            "Be specific and terse. Your output will be stored verbatim in a memory log "
            "and re-read by future analysts, so every word must earn its place."
        )

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
        benchmark_name: str = "SPY",
        holding_days: int = 5,
    ) -> str:
        """结合真实结果，对最终交易决策进行一次 LLM 反思。

        待结算决策完成结算时调用。最终交易决策已经汇总各分析师见解，
        因此无需额外提供市场上下文。``benchmark_name`` 是 Alpha 基准名称，
        美股如 ``"SPY"``，日股（``.T``）如 ``"^N225"``；
        尚未传入该参数的旧调用者默认使用 SPY。"""
        messages = [
            ("system", self._system_prompt(holding_days)),
            (
                "human",
                (
                    f"Raw return over {holding_days} trading days: {raw_return:+.1%}\n"
                    f"Alpha vs {benchmark_name}: {alpha_return:+.1%}\n\n"
                    f"Final Decision:\n{final_decision}"
                ),
            ),
        ]
        return self.quick_thinking_llm.invoke(messages).content
