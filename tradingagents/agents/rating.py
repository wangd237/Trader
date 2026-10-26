"""共用的五级评级体系与确定性规则解析器。

Research Manager、Portfolio Manager 和 Memory Log 均采用
Buy、Overweight、Hold、Underweight、Sell 五级评级。
统一解析以避免不同入口出现口径差异。
无法识别评级时，``extract_rating`` 返回 None，调用方使用
不可交易的 ``REVIEW``，而不是虚构一个 Hold（#1170）。"""

from __future__ import annotations

import re
import unicodedata

# 标准五级评级，按看多到看空排序。
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

# 模型输出无法识别评级时发出的标记，不代表可交易仓位。
# 此时需要人工复核或重新运行，不能自动降级为 Hold。
# 转换为评级枚举前应先使用 ``is_review`` 检查。
RATING_REVIEW = "REVIEW"

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# 兼容 Rating: X、rating - X、Rating — **X** 等标签形式，
# 支持 Markdown 强调及多种分隔符；必须匹配独立的 rating 单词，
# 避免把正文中的其他词组误判为评级标签。
_RATING_LABEL_RE = re.compile(r"(?<![a-z])rating\b[^:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(\w+)",
                              re.IGNORECASE)

# 识别 Portfolio Manager 自身决策的评级行，允许列表符号、标题、加粗
# 以及 Final Rating / Our rating 等写法。
# Consensus rating 或 Trader's rating 表示别人的评价，不是最终决策。
_OWN_QUALIFIER = r"(?:(?:final|our|overall|my|recommended|updated|revised|new|current)\s+)*"
_RATING_LINE_RE = re.compile(
    r"\s*(?P<item>(?:[-+*\u2022]|\d+[.)])\s+)?[\s*_#]*" + _OWN_QUALIFIER
    + r"rating[^\w:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(?P<value>\w+)",
    re.IGNORECASE,
)

# 排除仅展示等级说明的行，如 Rating Scale: Buy, ...。
_RATING_SCALE_RE = re.compile(r"rating\s*(scale|options|legend)", re.IGNORECASE)

def extract_rating(text: str) -> str | None:
    """从明确标签中解析五级评级；不存在时返回 None。

    先进行 NFKC 规范化，使全角标点与 ASCII 标点一致，
    再优先检查开头评级、决策自身的评级行及其他标签；
    如果候选结果冲突，则不猜测最终评级。"""
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)

    # 模型被要求在决策开头输出评级，因此优先考虑开头的评级行。
    # 列表中的评级也可能只是引用他人意见，应结合自身评级行判断。
    # 如果候选评级相互冲突、无法确认最终决策，则视为没有明确评级；
    # 等级刻度说明也不能当作决策（#1170）。
    lines = [line for line in norm.splitlines() if line.strip() and not _RATING_SCALE_RE.search(line)]
    first = next((line for line in lines if not line.lstrip().startswith("#")
                  or _RATING_LINE_RE.match(line)), "")
    m = _RATING_LINE_RE.match(first)
    opening = m if m and m.group("value").lower() in _RATING_SET else None
    if opening and not opening.group("item"):
        return opening.group("value").capitalize()

    own = [opening.group("value").capitalize()] if opening else []
    labels = []
    for line in lines:
        m = _RATING_LINE_RE.match(line)
        if m and not m.group("item") and m.group("value").lower() in _RATING_SET:
            own.append(m.group("value").capitalize())
        labels += [v.capitalize() for v in _RATING_LABEL_RE.findall(line) if v.lower() in _RATING_SET]
    # 没有明确标签就不应解析出评级；正文可能是在否定某种选择
    # （例如 not a Sell），直接匹配评级单词会产生错误结论。
    found = own or labels
    return found[0] if found and len(set(found)) == 1 else None


def parse_rating(text: str, default: str = RATING_REVIEW) -> str:
    """提取五级评级；不存在时返回 ``REVIEW``。

    供必须获得字符串的日志等调用者使用；默认不生成可交易评级。"""
    rating = extract_rating(text)
    return rating if rating is not None else default


def run_rating(final_state: dict) -> str:
    """读取完成运行的评级：优先使用 Portfolio Manager 的字段，
    否则从决策正文解析，以兼容没有 ``final_rating`` 的旧状态。"""
    return final_state.get("final_rating") or parse_rating(final_state.get("final_trade_decision", ""))


def is_review(signal: str) -> bool:
    """判断结果是否为不可交易的 REVIEW 标记（#1170）。"""
    return signal == RATING_REVIEW
