"""Single-call reflection: preserve the decision log prose and extract a reusable lesson."""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExtractedExperience(BaseModel):
    valuable: bool = False
    scenario: str = ""
    lesson: str = ""
    boundary: str = ""
    tags: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    reflection: str
    experience: ExtractedExperience = Field(default_factory=ExtractedExperience)


class Reflector:
    """Reflect on a settled decision with one model call."""

    def __init__(self, quick_thinking_llm: Any):
        self.quick_thinking_llm = quick_thinking_llm
        try:
            self.structured_llm = quick_thinking_llm.with_structured_output(ReflectionResult)
        except (NotImplementedError, AttributeError):
            self.structured_llm = None

    def _system_prompt(self, holding_days: int) -> str:
        return (
            "You are reviewing a past trading decision with a known outcome. "
            f"The realized result covers only {holding_days} trading days. "
            "Do not confuse a short-term observation with evidence for a longer thesis. "
            "Return a short reflection (2-4 sentences) discussing directional alpha, "
            "which part of the thesis was supported or challenged, and one lesson. "
            "Also extract at most one concise transferable experience: "
            "valuable, scenario, lesson, boundary and tags. "
            "Only set valuable=true if the lesson is specific, transferable, bounded, "
            "and supported by the stated evidence. Otherwise set valuable=false "
            "and leave other experience fields empty. "
            "Use short, lower_snake_case topic tags such as risk_control, "
            "relative_return, time_horizon, cash_flow, position_sizing, valuation, "
            "data_quality. Do not treat alpha alone as proof of causal correctness. "
            "Return the ReflectionResult schema when available."
        )

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
        benchmark_name: str = "SPY",
        holding_days: int = 5,
    ) -> ReflectionResult:
        """Produce both log reflection and reusable lesson without a second LLM call.

        Providers without structured output use a single plain-text call; only the
        historical reflection is retained in that compatibility fallback.
        """
        messages = [
            ("system", self._system_prompt(holding_days)),
            ("human",
             f"Raw return over {holding_days} trading days: {raw_return:+.1%}\n"
             f"Alpha vs {benchmark_name}: {alpha_return:+.1%}\n\n"
             f"Final Decision:\n{final_decision}"),
        ]
        if self.structured_llm is not None:
            try:
                result = self.structured_llm.invoke(messages)
                parsed = ReflectionResult.model_validate(result)
                if parsed.reflection.strip():
                    return parsed
                raise ValueError("empty structured reflection")
            except Exception as exc:
                logger.warning("Structured reflection failed; falling back to plain text: %s", exc)
        text = self.quick_thinking_llm.invoke(messages).content
        return ReflectionResult(reflection=str(text))
