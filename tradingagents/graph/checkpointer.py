"""为分析任务提供可恢复的 LangGraph 检查点支持。

每个 ticker 使用独立的 SQLite 数据库，避免不同标的之间的并发竞争。"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from tradingagents.dataflows.symbols import safe_ticker_component


def _db_path(data_dir: str | Path, ticker: str) -> Path:
    """返回指定 ticker 的 SQLite 检查点数据库路径。"""
    # 拒绝可能导致路径逃逸到检查点目录之外的 ticker。
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.db"


def thread_id(ticker: str, date: str, signature: str = "") -> str:
    """根据 ticker 和日期生成确定性的线程 ID。
    
    ``signature`` 包含影响图结构的运行选项，避免使用不同图配置恢复时
    复用旧检查点（#1089）；省略签名时仍兼容旧版线程 ID。"""
    base = f"{ticker.upper()}:{date}"
    if signature:
        base = f"{base}:{signature}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]


@contextmanager
def get_checkpointer(data_dir: str | Path, ticker: str) -> Generator[SqliteSaver, None, None]:
    """上下文管理器：返回使用当前 ticker 专属数据库的 SqliteSaver。"""
    db = _db_path(data_dir, ticker)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def checkpoint_step(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> int | None:
    """返回最新检查点的步骤编号；若不存在则返回 None。"""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return None
    tid = thread_id(ticker, date, signature)
    with get_checkpointer(data_dir, ticker) as saver:
        config = {"configurable": {"thread_id": tid}}
        cp = saver.get_tuple(config)
        if cp is None:
            return None
        return cp.metadata.get("step")


def clear_all_checkpoints(data_dir: str | Path) -> int:
    """删除全部检查点数据库，并返回删除的数据库数量。
    
    SQLite 会将已提交状态写入数据库旁的 ``-wal`` 和 ``-shm`` 文件；
    如果只删除 ``.db``，仍可能有检查点数据留在磁盘上。"""
    cp_dir = Path(data_dir) / "checkpoints"
    if not cp_dir.exists():
        return 0
    dbs = list(cp_dir.glob("*.db"))
    for db in dbs:
        for path in (db, *cp_dir.glob(f"{db.name}-*")):
            path.unlink(missing_ok=True)
    return len(dbs)


def clear_checkpoint(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> None:
    """通过删除对应线程的记录，清除指定 ticker 和日期的检查点。"""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return
    tid = thread_id(ticker, date, signature)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
