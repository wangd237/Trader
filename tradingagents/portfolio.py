"""决策 Agent 所看到的调用者持仓信息。

这是一次运行的可选输入，描述当前持仓、平均持仓价和可用现金。
缺少这些信息时，Agent 无法区分加仓现有满仓头寸和新开仓。
必须区分三种情况：有持仓、明确空仓、完全未提供持仓信息；
不能将“未提供”当作“空仓”，否则会虚构账户事实。

这里与券商无关：数量只是通用单位，币种由调用者标记，
不表示具体交易场所或下单执行路径。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class Position(BaseModel):
    ticker: str = Field(description="Instrument symbol, e.g. AAPL")
    quantity: float = Field(description="Signed units held; negative is short")
    average_price: float | None = Field(default=None, description="Average entry price per unit")


class PortfolioContext(BaseModel):
    cash: float | None = Field(default=None, description="Free cash available")
    currency: str | None = Field(default=None, description="Currency label for cash and prices")
    positions: list[Position] = Field(default_factory=list)

    def position_in(self, ticker: str) -> Position | None:
        return next((p for p in self.positions if p.ticker.upper() == ticker.strip().upper()), None)

    def render(self, ticker: str) -> str:
        """生成供决策 Agent 使用的持仓上下文，并将当前分析标的放在最前面。"""
        symbol = ticker.strip().upper()
        held = self.position_in(symbol)
        if held is None:
            lines = [f"- No current position in {symbol}"]
        else:
            price = f", average price {held.average_price:,.2f}" if held.average_price is not None else ""
            lines = [f"- Current position in {symbol}: {held.quantity:,.4g} units{price}"]
        if self.cash is not None:
            lines.append(f"- Cash available: {self.cash:,.2f}{' ' + self.currency if self.currency else ''}")
        others = [p for p in self.positions if p is not held]
        if others:
            lines.append("- Other positions: " + ", ".join(f"{p.ticker.upper()} {p.quantity:,.4g}" for p in others))
        return "Portfolio at the analysis date:\n" + "\n".join(lines)

    def fingerprint(self) -> str:
        """计算持仓的稳定摘要，使持仓变化后不会错误恢复旧运行。"""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def load_portfolio(path: str | Path) -> PortfolioContext:
    """读取持仓 JSON 文件；若失败则在运行前报错，而不是执行中途才失败。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return PortfolioContext.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"portfolio file {path} is not usable: {exc}") from exc
