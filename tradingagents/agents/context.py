"""Agent 共用的 Prompt 上下文：标的身份、输出语言与投资组合信息。"""

import functools
import logging
from collections.abc import Mapping
from typing import Any

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.vendors import ashare
from tradingagents.dataflows.vendors.yahoo.fundamentals import get_company_profile

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """返回指定输出语言对应的 Prompt 指令。

    默认英语时返回空字符串，不消耗额外 token。
    该指令应用于所有会输出到报告的 Agent，包括分析师、研究员、
    辩论员、研究经理、Trader 和投资组合经理，
    确保非英语运行生成完整的本地化报告，而不是中英混杂。"""
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    # 带标签的字段由程序解析，因此保持英文标签与取值。
    # 如果翻译评级标签，下游将退回到搜索自然语言正文，
    # 而包含否定的语句（如 “not a Sell”）可能被误判为 Sell（#1435）。
    return (
        f" Write your entire response in {lang}, except the labelled lines the format"
        f" asks for (the \"**Rating**:\" line, \"FINAL TRANSACTION PROPOSAL:\"):"
        f" keep their label and value in English, exactly as specified."
    )


def opponent_argument_or_opening(text: str, opponent: str) -> str:
    """返回对手最近一次论点；若为空则返回明确的开场标记。

    每轮辩论的首个发言者没有对手回复，如果直接将空文本插入
    “反驳对手”的 Prompt，模型可能虚构对方立场。
    改用“对手尚未发言”标记，让它正常提出自己的初始论点（#1176）。"""
    text = (text or "").strip()
    if text:
        return text
    return f"(The {opponent} has not spoken yet — open the debate with your own case.)"


def _clean_identity_value(value: Any) -> str | None:
    """返回去除首尾空白的字符串；为空或占位文本时返回 None。"""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


def resolve_instrument_identity(ticker: str) -> dict:
    """为 ticker 解析确定性的身份信息（公司名称、板块等）。

    这样可以防止分析师仅根据价格走势联想出错误行业，进而编造出
    另一家公司的身份，并将错误传递给全部下游 Agent（#814）。

    采用尽力而为策略：yfinance 不可用、被限流或不认识 ticker 时
    返回 ``{}``，由调用方退回仅含 ticker 的上下文，
    不会让分析在开始前失败。成功查询结果会在进程内缓存，
    失败查询下次仍会重试。
    身份解析与价格路径使用相同的标的映射，例如 ``XAUUSD`` → ``GC=F``（#983）。"""
    try:
        return _identity(ticker)
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}


@functools.lru_cache(maxsize=256)
def _identity(ticker: str) -> dict:
    """返回数据源中指定 ``ticker`` 的身份字段；查询失败时抛出异常。"""
    if ashare.is_ashare(ticker):
        return ashare.company_identity(ticker)
    info = get_company_profile(ticker)
    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
    trade_date: str | None = None,
) -> str:
    """描述准确的交易标的，确保 Agent 始终使用正确身份与 ticker。

    提供 ``identity``（由 :func:`resolve_instrument_identity` 确定性解析）
    后，会注入公司名称与业务分类，防止 Agent 根据价格图误认公司（#814）。

    该身份资料只有当下快照，没有历史版本。历史日期运行仅保留当前公司名
    用于区分公司，不会注入现在的板块、行业或交易所信息，
    因为它们在分析日期可能并不成立。"""
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "The tools serve this instrument; refer to it by this exact ticker in every report and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    identity = identity or {}
    name = identity.get("company_name") or identity.get("name")
    label = "Name" if is_crypto else "Company"
    details = []
    if is_historical(trade_date):
        if name:
            details.append(
                f"{label}: {name} (its current name, given only to identify it; "
                f"on {trade_date} it may have been named differently)"
            )
    else:
        if name:
            details.append(f"{label}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """返回当前运行的标的上下文。

    优先使用运行开始时解析并写入状态的身份信息
    （见 ``TradingAgentsGraph.resolve_instrument_context``）。
    若状态没有此信息（例如直接构造的状态或测试），则退回
    仅含 ticker 的上下文，不在 Graph 执行中途发起网络查询。"""
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def report_or_absent(text: str, source: str) -> str:
    """返回分析师报告；若没有报告则返回明确的缺失标记。

    未选择分析师、模型拒绝回答或返回空文本时，报告都可能为空。
    若直接在带标题的章节插入空文本，下游 Agent 可能把缺失信息
    臆测成具体结论，类似于空白对手论点导致虚构反驳的问题（#1176）。"""
    text = (text or "").strip()
    if text:
        return text
    return f"(No {source} report in this run: it is not available, not an empty finding.)"


def get_portfolio_context_from_state(state: Mapping[str, Any]) -> str:
    """返回调用方提供的投资组合信息，或明确说明未提供。

    没有持仓上下文不代表空仓；否则 Agent 会假设账户没有持仓
    并据此计算头寸规模，实际上调用者并未提供这种账户事实。"""
    context = state.get("portfolio_context")
    if isinstance(context, str) and context.strip():
        return context
    return (
        "Portfolio context: not provided. You do not know the caller's current "
        "holdings or cash, so do not assume a flat book; give direction and "
        "sizing guidance in terms the caller can apply to their own position."
    )
