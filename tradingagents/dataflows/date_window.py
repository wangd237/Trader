"""所有带日期的数据调用共享的时点约束：只能使用分析日期当时可获得的信息。

- ``as_of`` / ``as_of_window`` 将模型要求的日期或窗口限制在交易日之前，
  避免工具带着未来日期请求数据提供商。
- ``in_window`` 把新闻、StockTwits、Reddit 等事件限制在分析窗口内；
  时间统一为 UTC，结束日期的次日零点不包含在区间内。无日期记录
  仅在窗口延伸到当下时保留，以免回测使用无法确认是否来自未来的信息
  （#1126、#1220）。
- ``coverage_gap`` 将数据源未覆盖的窗口标为“不可用”，而不是“空结果”。
- ``withhold_live_profile`` 禁止历史运行读取今天的公司信息快照。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta


def to_utc(dt: datetime) -> datetime:
    """将 datetime 规范化为带时区的 UTC 时间；无时区值按 UTC 解释。"""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def in_window(pub_dt: datetime | None, start_dt: datetime, end_dt: datetime) -> bool:
    """判断记录是否处于左闭右开窗口 ``[start, end + 1 day)`` 中。

    ``pub_dt`` 为 None 表示没有日期；只有窗口延伸到当前时才保留。"""
    end = to_utc(end_dt)
    if pub_dt is not None:
        return to_utc(start_dt) <= to_utc(pub_dt) < end + timedelta(days=1)
    return end >= datetime.now(UTC) - timedelta(days=1)


def get_current_date() -> str:
    """返回今天的日期，格式为 YYYY-MM-DD。"""
    return date.today().strftime("%Y-%m-%d")


def is_historical(run_date) -> bool:
    """判断本次运行是否为历史日期，以识别只能获取当前快照的数据是否存在未来信息。"""
    return bool(run_date) and str(run_date) < get_current_date()


def coverage_gap(
    dates, start_date: str, end_date: str, source: str, subject: str
) -> str | None:
    """为数据源尚未完整覆盖的窗口生成占位提示；已覆盖则返回 None。

    Yahoo 新闻、Reddit 和 StockTwits 无论请求什么时间范围，
    通常只返回最新数据；若某个窗口根本未被它们覆盖，
    就不能把“没有找到”当成该窗口确实“没有新闻”。
    只有覆盖范围包含窗口首日，且窗口不晚于今天，空结果才代表真实缺失。

    ``dates`` 包含返回记录的时间戳，对固定回溯窗口的数据源
    还包含回溯起点。仅当数据按时间倒序且连续返回时，最早记录
    才能作为覆盖边界；合并或按相关性排序的结果不提供 dates，
    此时只确认当前时点的覆盖。"""
    now = datetime.now(UTC)
    oldest = min((to_utc(d) for d in dates if d is not None), default=now)
    if datetime.strptime(end_date, "%Y-%m-%d").date() > now.date():
        reason = "the window extends past today"
    elif oldest.date() > datetime.strptime(start_date, "%Y-%m-%d").date():
        reason = "it only serves recent items"
    else:
        return None
    return f"<{source} unavailable for {start_date}..{end_date}: {reason}, so this is not an absence of {subject}>"


def _parse(date: str | None) -> datetime | None:
    try:
        return datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def as_of(requested: str | None, trade_date: str) -> str | None:
    """确定工具实际服务的日期：允许使用模型传入日期，但不能晚于分析日期。

    模型可能省略日期或填入今天，而非历史分析日期，从而绕开工具内部的
    时点限制。在 Graph 外直接调用、``trade_date`` 为空时，原样透传请求。"""
    if not trade_date:
        return requested
    parsed = _parse(requested)
    return requested if parsed is not None and parsed <= _parse(trade_date) else trade_date


def as_of_window(start_date: str, end_date: str, trade_date: str) -> tuple[str, str]:
    """返回 ``[start, end]`` 窗口，并将结束日期限制在分析日期之前。

    若整个窗口位于分析日期之后，则保持长度并整体前移。
    无法解析的结束日期按分析日期处理；起始日期无法解析时
    不能猜测窗口，应抛出 ValueError 供调用者反馈（#1476）。"""
    start, old_end = _parse(start_date), _parse(end_date)
    if start is None:
        raise ValueError(f"start_date {start_date!r} is not a date; give it as YYYY-MM-DD")
    end = as_of(end_date, trade_date)
    if end == end_date or start <= _parse(end):
        return start_date, end
    span = (old_end - start) if old_end is not None and old_end >= start else timedelta(0)
    return f"{_parse(end) - span:%Y-%m-%d}", end


def withhold_live_profile(as_of_date: str | None, label: str) -> str | None:
    """历史运行不能使用当前公司概况时返回提示，否则返回 None。

    yfinance ``Ticker.info``、Alpha Vantage ``OVERVIEW`` 等公司概况接口
    没有历史版本信息；公司名称、行业和板块都可能因更名或重分类变化。
    将今天的资料用于历史日期会产生决策后的信息泄漏（#1300）。
    所有基本面提供商都遵守此规则，不能通过切换数据源绕过。"""
    if not is_historical(as_of_date):
        return None
    return (
        f"# Company Fundamentals for {label}\n"
        f"# Point-in-time as of: {as_of_date}\n\n"
        f"Profile fundamentals are withheld for this date. This vendor serves "
        f"only present-day values with no historical vintage: market "
        f"cap, valuation multiples, the 52-week range and TTM income move with "
        f"today's quote, and even the name, sector and industry reflect today "
        f"rather than {as_of_date} (companies rename and get reclassified). "
        f"Serving them would put post-decision information into a {as_of_date} "
        f"analysis. Statements filed by {as_of_date} are served as filed where "
        f"the filing date is known (SEC EDGAR, for US filers)."
    )


def withhold_undisclosed_trades(as_of_date: str | None, label: str) -> str | None:
    """内部人交易没有披露日期且不能安全用于历史分析时返回提示，否则返回 None。

    Yahoo 与 Alpha Vantage 等提供商可能只记录交易发生时间，
    无法确认何时公开；Form 4 最晚可在两个工作日后提交。
    因此历史运行不能直接使用可能尚未披露的交易记录。"""
    if not is_historical(as_of_date):
        return None
    return (
        f"# Insider Transactions for {label}\n"
        f"# Point-in-time as of: {as_of_date}\n\n"
        f"Insider transactions are withheld for this date. This vendor dates a trade "
        f"by when it happened and reports no filing date, so it cannot show which "
        f"trades were public on {as_of_date}."
    )


def withhold_undated_statements(as_of_date: str | None, label: str, title: str) -> str | None:
    """财务报表缺少披露日期且不能用于历史分析时返回提示，否则返回 None。

    Yahoo、Alpha Vantage 的报表日期通常表示所属财务期间，
    而非实际公开时间；公司可能在期末数周后才披露。
    在此间隔内运行回测会看到当时尚不存在的数据。
    SEC EDGAR 记录了每次披露日期，因此可以按实际披露时间
    向美国上市公司的历史分析提供报表。"""
    if not is_historical(as_of_date):
        return None
    return (
        f"# {title} for {label}\n"
        f"# Point-in-time as of: {as_of_date}\n\n"
        f"{title} data is withheld for this date. This vendor dates a statement by "
        f"the period it covers and reports no filing date, so it cannot show which "
        f"figures were public on {as_of_date}. SEC EDGAR serves US filers as filed."
    )
