"""验证从交易决策中提取五级评级的规则解析器。

Portfolio Manager 的渲染报告有固定的 ``**Rating**: X`` 标题，
因此可以用确定性代码解析，无需再调用一次模型。"""

import pytest

from tradingagents.agents.rating import RATING_REVIEW, RATINGS_5_TIER, extract_rating, parse_rating

# ---------------------------------------------------------------------------
# Heuristic parser
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseRating:
    def test_explicit_label_buy(self):
        assert parse_rating("Rating: Buy\nReasoning here.") == "Buy"

    def test_explicit_label_overweight(self):
        assert parse_rating("Rating: Overweight\nDetails.") == "Overweight"

    def test_explicit_label_with_markdown_bold_value(self):
        # 回归测试：允许 Rating: **Sell** 等 Markdown 强调写法。
        assert parse_rating("Rating: **Sell**\nExit immediately.") == "Sell"

    def test_explicit_label_with_markdown_bold_label(self):
        assert parse_rating("**Rating**: Underweight\nTrim exposure.") == "Underweight"

    def test_rendered_pm_markdown_shape(self):
        # render_pm_decision 实际输出的格式必须始终可解析。
        text = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Enter at $189-192, 6% portfolio cap.\n\n"
            "**Investment Thesis**: AI capex cycle intact; institutional flows constructive."
        )
        assert parse_rating(text) == "Buy"

    def test_explicit_label_wins_over_prose_with_markdown(self):
        text = (
            "The buy thesis is weakened by guidance.\n"
            "Rating: **Sell**\n"
            "Exit before earnings."
        )
        assert parse_rating(text) == "Sell"

    def test_no_rating_is_flagged_for_review_not_defaulted(self):
        # 无法识别的决策不等于 Hold，强行记录 Hold 会编造不存在的评级。
        assert parse_rating("No clear directional signal at this time.") == RATING_REVIEW

    def test_no_rating_custom_default(self):
        assert parse_rating("Plain prose.", default="Underweight") == "Underweight"

    def test_all_five_tiers_recognised(self):
        for r in RATINGS_5_TIER:
            assert parse_rating(f"Rating: {r}") == r

    def test_fullwidth_colon_is_parsed_not_reviewed(self):
        # `Rating：Overweight` 使用全角冒号，也应成功解析而不是进入复核（#1170）。
        assert parse_rating("Rating：Overweight\n理由はこちら。") == "Overweight"


@pytest.mark.unit
class TestExtractRating:
    def test_returns_none_when_absent(self):
        assert extract_rating("No directional call here.") is None
        assert extract_rating("") is None

    def test_whole_word_only(self):
        # 不能匹配其他单词内部的评级子串。
        assert extract_rating("The buyer was holding shares.") is None

    def test_parse_rating_defaults_to_review(self):
        # Memory Log 应将无法识别的决策标为 REVIEW，不能写入可交易评级。
        assert parse_rating("No rating here.") == RATING_REVIEW
        assert parse_rating("No rating here.", default="Underweight") == "Underweight"
