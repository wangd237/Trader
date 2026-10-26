"""记忆日志：以追加写入的 Markdown 形式记录每笔决策，并在结算后补充实际结果。"""

import contextlib
import re
from pathlib import Path

from tradingagents.agents.rating import parse_rating
from tradingagents.dataflows.files import locked


class TradingMemoryLog:
    """只追加写入的交易决策与反思 Markdown 日志。"""

    # HTML 注释标记不会自然出现在 LLM 正文中，适合作为可靠的硬分隔符。
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    # 预编译正则表达式，避免每次 load_entries() 都重新编译。
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._log_path = None
        path = cfg.get("memory_log_path")
        if path:
            self._log_path = Path(path).expanduser()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # 可选的已结算记录数量上限；None 表示禁用轮转清理。
        self._max_entries = cfg.get("memory_log_max_entries")

    # --- 写入：记录每次运行产生的决策 ---

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
        rating: str | None = None,
    ) -> None:
        """在 propagate() 结束时追加一条待结算记录，不调用 LLM。
        
        若调用者传入 ``rating``，使用该决策评级；否则从决策文本中解析评级。"""
        if not self._log_path:
            return
        with locked(self._log_path):
            # 幂等保护：直接扫描原始文本而非完整解析。只要相同 ticker 与日期
            # 已有记录（无论待结算还是已结算），就阻止重复写入；
            # 否则重复运行可能把同一决策多次计入历史上下文和日志聚合结果。
            if self._log_path.exists():
                raw = self._log_path.read_text(encoding="utf-8")
                for line in raw.splitlines():
                    if line.startswith(f"[{trade_date} | {ticker} |") and line.endswith("]"):
                        return
            rating = rating or parse_rating(final_trade_decision)
            tag = f"[{trade_date} | {ticker} | {rating} | pending]"
            entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry)

    def settling(self, wait: bool = True):
        """跨线程、跨进程保证同一时间只有一轮结算持有日志锁。
        
        结算会读取待处理记录，对每笔决策调用模型生成反思并写回。
        并行结算可能对同一笔决策重复付费，因此使用独立于单次写入锁的
        专用结算锁。返回是否成功持有锁；设置 ``wait=False`` 时，
        不等待其他正在执行的结算轮次。"""
        if not self._log_path:
            return contextlib.nullcontext(True)
        return locked(f"{self._log_path}.settle", wait=wait)

    # --- 读取 ---

    def load_entries(self) -> list[dict]:
        """解析日志中的全部记录，并返回字典列表。"""
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        entries = []
        for raw in raw_entries:
            parsed = self._parse_entry(raw)
            if parsed:
                entries.append(parsed)
        return entries

    def get_pending_entries(self) -> list[dict]:
        """返回 outcome:pending 的记录，供结算流程处理。"""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_past_context(
        self, ticker: str, n_same: int = 5, n_cross: int = 3, as_of: str | None = None
    ) -> str:
        """生成格式化的历史经验字符串，供注入 Agent Prompt。
        
        指定 ``as_of``（yyyy-mm-dd）时，只包含该日期之前已经知道结果的经验；
        记录必须包含 ``resolved:...`` 结算日期，且该日期不晚于 ``as_of``。
        这可防止历史分析或回测从当时尚未发生的结果中学习（#1251）。
        ``as_of=None`` 关闭过滤，以兼容实时运行和迁移前的旧日志记录。"""
        entries = [e for e in self.load_entries() if not e.get("pending")]
        if as_of is not None:
            entries = [e for e in entries if e.get("resolved") and e["resolved"] <= as_of]
        if not entries:
            return ""

        same, cross = [], []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker and len(same) < n_same:
                same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        if not same and not cross:
            return ""

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    # --- 结算：记录决策结果与反思 ---

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str,
        resolution_date: str | None = None,
    ) -> bool:
        """以原子写入方式更新待结算标记，并追加 REFLECTION 章节。
        
        返回记录是否成功更新；若记录已经不再待结算（其他任务先完成），
        则返回 False。
        
        找到与 (trade_date, ticker) 匹配的第一条待结算记录，
        写入收益率和 ``resolution_date``（实际结果变得可知的日期），
        并追加反思章节。通过临时文件加 os.replace() 原子替换，
        防止写入中途崩溃破坏日志。"""
        if not self._log_path or not self._log_path.exists():
            return False
        with locked(self._log_path):
            text = self._log_path.read_text(encoding="utf-8")
            blocks = text.split(self._SEPARATOR)

            pending_prefix = f"[{trade_date} | {ticker} |"
            raw_pct = f"{raw_return:+.1%}"
            alpha_pct = f"{alpha_return:+.1%}"

            updated = False
            new_blocks = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    new_blocks.append(block)
                    continue

                lines = stripped.splitlines()
                tag_line = lines[0].strip()

                if (
                    not updated
                    and tag_line.startswith(pending_prefix)
                    and tag_line.endswith("| pending]")
                ):
                    fields = [f.strip() for f in tag_line[1:-1].split("|")]
                    rating = fields[2]
                    new_tag = self._resolved_tag(
                        trade_date, ticker, rating, raw_pct, alpha_pct, holding_days, resolution_date
                    )
                    rest = "\n".join(lines[1:])
                    new_blocks.append(
                        f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{reflection}"
                    )
                    updated = True
                else:
                    new_blocks.append(block)

            if not updated:
                return False

            new_blocks = self._apply_rotation(new_blocks)
            new_text = self._SEPARATOR.join(new_blocks)
            tmp_path = self._log_path.with_suffix(".tmp")
            tmp_path.write_text(new_text, encoding="utf-8")
            tmp_path.replace(self._log_path)
            return True

    def batch_update_with_outcomes(self, updates: list[dict]) -> None:
        """在一次文件读取和原子写入中批量更新多条决策结果。
        
        updates 中每个元素都必须提供 ticker、trade_date、
        raw_return、alpha_return、holding_days 和 reflection。"""
        if not self._log_path or not self._log_path.exists() or not updates:
            return
        with locked(self._log_path):
            text = self._log_path.read_text(encoding="utf-8")
            blocks = text.split(self._SEPARATOR)

            update_map = {(u["trade_date"], u["ticker"]): u for u in updates}

            new_blocks = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    new_blocks.append(block)
                    continue

                lines = stripped.splitlines()
                tag_line = lines[0].strip()

                matched = False
                for (trade_date, ticker), upd in list(update_map.items()):
                    pending_prefix = f"[{trade_date} | {ticker} |"
                    if tag_line.startswith(pending_prefix) and tag_line.endswith("| pending]"):
                        fields = [f.strip() for f in tag_line[1:-1].split("|")]
                        rating = fields[2]
                        raw_pct = f"{upd['raw_return']:+.1%}"
                        alpha_pct = f"{upd['alpha_return']:+.1%}"
                        new_tag = self._resolved_tag(
                            trade_date, ticker, rating, raw_pct, alpha_pct,
                            upd["holding_days"], upd.get("resolution_date"),
                        )
                        rest = "\n".join(lines[1:])
                        new_blocks.append(
                            f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{upd['reflection']}"
                        )
                        del update_map[(trade_date, ticker)]
                        matched = True
                        break

                if not matched:
                    new_blocks.append(block)

            new_blocks = self._apply_rotation(new_blocks)
            new_text = self._SEPARATOR.join(new_blocks)
            tmp_path = self._log_path.with_suffix(".tmp")
            tmp_path.write_text(new_text, encoding="utf-8")
            tmp_path.replace(self._log_path)

    # --- 辅助方法 ---

    @staticmethod
    def _resolved_tag(
        trade_date, ticker, rating, raw_pct, alpha_pct, holding_days, resolution_date
    ) -> str:
        """生成已结算记录标记，同时记录真实结果可知的日期。
        
        ``resolution_date`` 即计算收益率时使用的最后一条价格记录日期，
        后续运行据此执行时点过滤（#1251）。无法取得时省略，
        保持兼容旧版六字段标记。"""
        tag = f"[{trade_date} | {ticker} | {rating} | {raw_pct} | {alpha_pct} | {holding_days}d"
        if resolution_date:
            tag += f" | resolved:{resolution_date}"
        return tag + "]"

    def _apply_rotation(self, blocks: list[str]) -> list[str]:
        """已结算记录超过 max_entries 时，删除最早的已结算块。
        
        待结算块始终保留，因为它们仍有工作未完成。
        若禁用轮转或没有超过上限，则原样返回 ``blocks``。"""
        if not self._max_entries or self._max_entries <= 0:
            return blocks

        # 根据标记行判断每个记录块的 (是否保留, 是否已结算)。
        decisions = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                decisions.append((block, False))
                continue
            tag_line = stripped.splitlines()[0].strip()
            is_resolved = (
                tag_line.startswith("[")
                and tag_line.endswith("]")
                and not tag_line.endswith("| pending]")
            )
            decisions.append((block, is_resolved))

        resolved_count = sum(1 for _, r in decisions if r)
        if resolved_count <= self._max_entries:
            return blocks

        to_drop = resolved_count - self._max_entries
        kept: list[str] = []
        for block, is_resolved in decisions:
            if is_resolved and to_drop > 0:
                to_drop -= 1
                continue
            kept.append(block)
        return kept

    def _parse_entry(self, raw: str) -> dict | None:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None
        fields = [f.strip() for f in tag_line[1:-1].split("|")]
        if len(fields) < 4:
            return None
        # 可选的末尾 ``resolved:YYYY-MM-DD`` 字段记录结果变得可知的日期，
        # 供历史时点过滤使用（#1251）。
        resolved = None
        for f in fields[6:]:
            if f.startswith("resolved:"):
                resolved = f[len("resolved:"):].strip()
        entry = {
            "date": fields[0],
            "ticker": fields[1],
            "rating": fields[2],
            "pending": fields[3] == "pending",
            "raw": fields[3] if fields[3] != "pending" else None,
            "alpha": fields[4] if len(fields) > 4 else None,
            "holding": fields[5] if len(fields) > 5 else None,
            "resolved": resolved,
        }
        body = "\n".join(lines[1:]).strip()
        decision_match = self._DECISION_RE.search(body)
        reflection_match = self._REFLECTION_RE.search(body)
        entry["decision"] = decision_match.group(1).strip() if decision_match else ""
        entry["reflection"] = reflection_match.group(1).strip() if reflection_match else ""
        return entry

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"
