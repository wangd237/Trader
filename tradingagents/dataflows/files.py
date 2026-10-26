"""并发写入文件的辅助函数：保证完整替换文件，并让同一时刻只有一个写入者。"""

import errno
import logging
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def replace_file(path, write: Callable[[str], None]) -> None:
    """先写入同目录下唯一命名的临时文件，再将它移动到 ``path``。

    读取者只会看到旧文件或新文件，不会看到不完整的内容；
    同时写入同一路径的任务也不会共用临时文件。
    回调 ``write`` 接收临时路径并创建文件，沿用常规文件权限。
    由于这里写的是缓存文件，如果 Windows 上有读取者占用旧文件，
    则跳过本次替换，保留原文件。"""
    path = Path(path)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(str(temp))
        os.replace(temp, path)
    except PermissionError as exc:
        temp.unlink(missing_ok=True)
        logger.warning("Kept the cached %s; it is in use (%s)", path.name, exc)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


@contextmanager
def locked(path, wait: bool = True) -> Iterator[bool]:
    """跨线程与进程确保同一时间只有一个写入者持有 ``path``。

    用于读取、修改、再写回的场景：若不加锁，两个写入者可能
    都读到相同旧内容，后一次写入会覆盖前一次更新。
    锁位于 ``path`` 旁的 ``.lock`` 文件。返回是否已获得锁；
    ``wait=False`` 时不会等待其他写入者，未获取锁也会进入代码块，
    但此时不持有锁。"""
    with open(f"{path}.lock", "a+b") as handle:
        hold = _hold_windows if os.name == "nt" else _hold_posix
        with hold(handle, wait) as acquired:
            yield acquired


@contextmanager
def _hold_posix(handle, wait: bool = True) -> Iterator[bool]:
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _hold_windows(handle, wait: bool = True) -> Iterator[bool]:
    import msvcrt

    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            break
        except OSError as exc:
            if not wait and exc.errno in (errno.EACCES, errno.EDEADLOCK):
                yield False   # another writer holds it
                return
            # 当其他写入者占有锁超过约十秒时，LK_LOCK 会放弃等待；
            # 其他错误不能视为普通的锁等待。
            if exc.errno != errno.EDEADLOCK:
                raise
    try:
        yield True
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
