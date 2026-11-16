"""Small, auditable Markdown store for reusable lessons extracted from settled decisions."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from tradingagents.dataflows.files import locked
from tradingagents.memory.reflection import ExtractedExperience

logger = logging.getLogger(__name__)

# Only a modest controlled vocabulary is needed for the initial rule-based search.
TAG_ALIASES = {
    "risk_control": ("降险", "减仓", "回撤", "止损", "下行风险", "risk control", "reduce exposure", "drawdown", "stop loss", "downside risk"),
    "relative_return": ("相对收益", "超额收益", "跑赢基准", "alpha", "benchmark", "relative return", "outperformance"),
    "time_horizon": ("观察期", "短期", "长期", "持有期", "时间窗口", "5日", "5天", "horizon", "short-term", "long-term", "holding period"),
    "cash_flow": ("现金流", "经营现金", "自由现金", "cash flow", "free cash"),
    "valuation": ("估值", "市盈率", "市净率", "valuation", "pe ratio", "price-to-earnings"),
    "earnings": ("盈利", "利润", "财报", "业绩", "earnings", "profit", "revenue"),
    "position_sizing": ("仓位", "仓储", "头寸", "position sizing", "allocation", "exposure", "portfolio weight"),
    "data_quality": ("数据缺失", "缺乏证据", "数据质量", "data quality", "missing data", "stale data"),
    "market_regime": ("市场环境", "宏观环境", "市场周期", "market regime", "macro conditions"),
    "liquidity": ("流动性", "成交量", "liquidity", "trading volume"),
    "event_risk": ("事件风险", "政策变化", "财报公布", "event risk", "earnings announcement", "regulatory"),
    "uncertainty": ("不确定性", "信心不足", "证据不足", "uncertainty", "insufficient evidence", "low confidence"),
}
STOPWORDS = frozenset({
    "with", "from", "this", "that", "were", "have", "been", "would", "should",
    "their", "which", "about", "after", "before", "current", "report", "history",
    "investment", "analysis", "decision", "trader", "market", "manager", "stock",
})
_JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def _terms(text: str) -> set[str]:
    """Limited English tokens and domain phrases; Chinese phrases use an explicit vocabulary."""
    lower = text.lower()
    terms = {
        w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", lower)
        if w not in STOPWORDS and len(w) <= 40
    }
    terms.update(alias for aliases in TAG_ALIASES.values() for alias in aliases if alias in lower)
    return terms


def _context_tags(text: str) -> set[str]:
    lower = text.lower()
    return {
        tag for tag, aliases in TAG_ALIASES.items()
        if any(alias in lower for alias in aliases)
    }


def build_experience_query(state: dict) -> str:
    """Read the finished plans and risk discussion, never the whole growing message history."""
    risk = state.get("risk_debate_state") or {}
    return "\n".join((
        str(state.get("investment_plan") or "")[:1500],
        str(state.get("trader_investment_plan") or "")[:1500],
        str(risk.get("history") or "")[-1800:],
    ))


def format_experiences(entries: list[dict]) -> str:
    """Bound prompt size and clearly mark retrieved experience as fallible context."""
    blocks = []
    for e in entries[:3]:
        blocks.append(
            f"[{e['id']} | {e['ticker']} | {e['trade_date']} | resolved:{e['resolved_at']}]\n"
            f"Scenario: {e['scenario'][:240]}\n"
            f"Lesson: {e['lesson'][:320]}\n"
            f"Boundary: {e['boundary'][:240]}"
        )
    return "\n\n".join(blocks)


class ExperienceStore:
    """Separate Markdown view of transferable lessons, keyed by the source decision."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        configured = config.get("experience_memory_path")
        decision_log = config.get("memory_log_path")
        self._path = (
            Path(configured).expanduser()
            if configured else Path(decision_log).expanduser().with_name("experiences.md")
            if decision_log else None
        )
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def load_entries(self) -> list[dict]:
        if not self._path or not self._path.exists():
            return []
        entries: list[dict] = []
        for match in _JSON_BLOCK.finditer(self._path.read_text(encoding="utf-8")):
            try:
                item = json.loads(match.group(1))
                if isinstance(item, dict) and item.get("source_id"):
                    entries.append(item)
            except (ValueError, TypeError):
                logger.warning("Skipping malformed experience entry in %s", self._path)
        return entries

    def store(
        self, *, ticker: str, trade_date: str, resolved_at: str | None,
        experience: ExtractedExperience,
    ) -> bool:
        """Validate useful lessons and append once per decision under a file lock."""
        if not self._path or not experience.valuable:
            return False
        scenario = experience.scenario.strip()
        lesson = experience.lesson.strip()
        boundary = experience.boundary.strip()
        tags = sorted(set(experience.tags))
        if not all((scenario, lesson, boundary, tags)):
            return False
        if any(len(field) > limit for field, limit in (
            (scenario, 240), (lesson, 320), (boundary, 240),
        )):
            return False
        source_id = f"{ticker}|{trade_date}"
        item = {
            "id": "EXP-" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12].upper(),
            "source_id": source_id,
            "ticker": ticker,
            "trade_date": trade_date,
            "resolved_at": resolved_at,
            "scenario": scenario,
            "lesson": lesson,
            "boundary": boundary,
            "tags": tags,
        }
        with locked(self._path):
            if any(e.get("source_id") == source_id for e in self.load_entries()):
                return False
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n## {item['id']}\n\n"
                    "```json\n"
                    + json.dumps(item, ensure_ascii=False, indent=2)
                    + "\n```\n"
                )
        return True

    def retrieve(self, query: str, *, as_of: str | None = None, top_k: int = 3) -> list[dict]:
        """Cross-ticker tag/keyword retrieval with a historical point-in-time gate."""
        if not query.strip() or top_k <= 0:
            return []
        query_tags, query_terms = _context_tags(query), _terms(query)
        ranked = []
        for item in self.load_entries():
            resolved_at = item.get("resolved_at")
            if as_of is not None and (not resolved_at or resolved_at > as_of):
                continue
            item_tags = set(item.get("tags") or [])
            content = " ".join(str(item.get(field) or "") for field in ("scenario", "lesson", "boundary"))
            common_tags = query_tags & item_tags
            common_terms = query_terms & _terms(content)
            # Two independent content hits can also recall lessons with imperfect labels.
            if not common_tags and len(common_terms) < 2:
                continue
            score = 3 * len(common_tags) + 0.25 * len(common_terms)
            ranked.append((score, str(resolved_at or ""), str(item.get("id") or ""), item))
        # Higher relevance wins; on a tie, prefer the more recently resolved lesson.
        # ID provides a deterministic final tie-breaker.
        ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        return [item for _, _, _, item in ranked[:top_k]]
