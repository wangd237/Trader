"""向终端用户展示更新信息，包括新版本和 Tauric Research 公告。

后台线程最多每天一次查询 GitHub 最新发布版本及公告服务，
并将结果写入轻量缓存。启动时直接读取缓存，因此不会等待网络；
最新消息在下次启动时显示。请求不会发送可识别用户身份的信息。
无人值守、脚本或 CI 运行不会连接这些服务。
设置 TRADINGAGENTS_NO_UPDATE_CHECK=1 可以完全禁用检查。"""

import importlib.metadata
import json
import os
import re
import shlex
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from packaging.version import InvalidVersion, Version

from tradingagents.dataflows.files import replace_file
from tradingagents.default_config import _TRADINGAGENTS_HOME

RELEASES_URL = "https://api.github.com/repos/TauricResearch/TradingAgents/releases/latest"
NOTICES_URL = "https://api.tauric.ai/v1/announcements"
CACHE = Path(_TRADINGAGENTS_HOME) / "update_check.json"
CHECK_EVERY = 24 * 3600
TIMEOUT = 2.0
SUMMARY_LIMIT = 160
OPT_OUT = "TRADINGAGENTS_NO_UPDATE_CHECK"
# C0/C1 控制字符替换为空格；删除不可见空格和文字方向控制符。
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
INVISIBLE = re.compile(r"[\u061c\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def enabled() -> bool:
    """判断当前是否为交互式终端，且用户未关闭更新检查。"""
    at_a_terminal = bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    return at_a_terminal and not os.environ.get("CI") and not os.environ.get(OPT_OUT)


# --- 缓存与检查 -------------------------------------------------------

def read_cache() -> dict:
    """读取缓存，只保留当前版本认可的字段结构。"""
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    shapes = {"checked_at": (int, float), "latest": dict, "notices": list}
    return {key: data[key] for key, shape in shapes.items() if isinstance(data.get(key), shape)}


def due(cache: dict, now: float) -> bool:
    checked = cache.get("checked_at")
    return not isinstance(checked, (int, float)) or not 0 <= now - checked < CHECK_EVERY


def summary_of(body: str) -> str:
    """提取发布说明的首段并合并为单行摘要（显示时会清洗内容）。"""
    first = re.split(r"\n\s*\n", (body or "").strip(), maxsplit=1)[0]
    text = " ".join(first.split())
    if len(text) > SUMMARY_LIMIT:
        text = text[: SUMMARY_LIMIT - 3].rsplit(" ", 1)[0] + "..."
    return text


def parse_notices(payload) -> list | None:
    """从公告响应中提取定向通知；不存在通知列表时返回 None。"""
    notices = payload.get("notices") if isinstance(payload, dict) else None
    return notices if isinstance(notices, list) else None


def _get_json(url: str):
    response = requests.get(url, timeout=TIMEOUT)
    return response.json() if response.ok else None


def refresh(now: float | None = None) -> None:
    """查询最新版本及公告，并记录结果。不抛出异常，也不打印消息；
    检查失败时保留已有缓存，当天不再重试。"""
    now = time.time() if now is None else now
    cache = read_cache()
    cache["checked_at"] = now
    try:
        release = _get_json(RELEASES_URL)
        if isinstance(release, dict) and release.get("tag_name"):
            cache["latest"] = {
                "version": str(release["tag_name"]).removeprefix("v"),
                "url": release.get("html_url"),
                "summary": summary_of(release.get("body") or ""),
            }
    except Exception:
        pass
    try:
        notices = parse_notices(_get_json(NOTICES_URL))
        if notices is not None:
            cache["notices"] = notices
    except Exception:
        pass
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        replace_file(CACHE, lambda temp: Path(temp).write_text(json.dumps(cache), encoding="utf-8"))
    except Exception:
        pass


def start_background_refresh() -> None:
    """到达检查时间后，在守护线程中刷新缓存。"""
    if due(read_cache(), time.time()):
        threading.Thread(target=refresh, name="tradingagents-update-check", daemon=True).start()


# --- 更新信息展示 -------------------------------------------------------

def _version(text) -> Version | None:
    try:
        return Version(str(text))
    except (InvalidVersion, TypeError):
        return None


def _plain(text) -> str:
    return " ".join(CONTROL.sub(" ", INVISIBLE.sub("", str(text or ""))).split())


def _applies(notice: dict, running: Version, today: date) -> bool:
    """判断公告是否针对当前版本且尚未过期；
    无法解析版本限制或日期时将其隐藏。
    公告显示至其 ``expires`` 日期（UTC）结束。"""
    low, high = notice.get("min_version"), notice.get("max_version")
    if low is not None and ((floor := _version(low)) is None or running < floor):
        return False
    if high is not None and ((ceiling := _version(high)) is None or running > ceiling):
        return False
    expires = notice.get("expires")
    if expires is None:
        return True
    try:
        return isinstance(expires, str) and bool(DATE.fullmatch(expires)) and date.fromisoformat(expires) >= today
    except ValueError:
        return False


def messages(cache: dict, running: str, upgrade: Callable[[], str | None] = lambda: None) -> list[str]:
    """生成要显示的文本行：新版本及升级命令（仅当有新版本时询问
    ``upgrade()``），然后显示适用于当前版本的公告。"""
    current = _version(running)
    if current is None:
        return []
    lines = []
    latest = cache.get("latest") or {}
    newest = _version(latest.get("version"))
    if newest is not None and newest > current:
        lines.append(f"TradingAgents {_plain(latest['version'])} is available (you have {_plain(running)}): "
                     f"{_plain(latest.get('summary'))}")
        if latest.get("url"):
            lines.append(_plain(latest["url"]))
        command = upgrade()
        if command:
            lines.append(f"Upgrade: {_plain(command)}")
    today = datetime.now(UTC).date()
    for notice in cache.get("notices") or []:
        if not (isinstance(notice, dict) and isinstance(notice.get("message"), str) and _plain(notice["message"])
                and _applies(notice, current, today)):
            continue
        url = notice.get("url")
        link = url if isinstance(url, str) and url.lower().startswith(("http://", "https://")) else None
        lines.append(_plain(notice["message"]) + (f" {_plain(link)}" if link else ""))
    return lines


def upgrade_command(dist=None) -> str | None:
    """根据 pip 或 uv 的安装记录构造升级命令；无法确定时返回 None。"""
    if os.environ.get("TRADINGAGENTS_IN_DOCKER"):
        return "git pull && docker compose build (on the host)"
    if os.name == "nt":                    # the commands below are POSIX shell
        return None
    try:
        dist = dist or importlib.metadata.distribution("tradingagents")
        direct = json.loads(dist.read_text("direct_url.json") or "null")
        installer = (dist.read_text("INSTALLER") or "").strip()
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return None
    if not isinstance(direct, dict) or not direct.get("url"):
        return None
    python = shlex.quote(sys.executable)
    install = f"uv pip install --python {python}" if installer == "uv" else f"{python} -m pip install"
    if "vcs_info" in direct:
        return f"{install} -U git+{direct['url']}"
    parsed = urlparse(direct["url"])
    if parsed.scheme != "file":
        return None
    clone = Path(unquote(parsed.path))
    if not (clone / ".git").exists():
        return None
    editable = (direct.get("dir_info") or {}).get("editable")
    target = shlex.quote(str(clone))
    return f"git -C {target} pull && {install} {'-e ' if editable else ''}{target}"


def show(console) -> None:
    """在欢迎面板下方打印缓存中的更新信息。"""
    from rich.panel import Panel
    from rich.text import Text

    from tradingagents import __version__

    lines = messages(read_cache(), running=__version__, upgrade=upgrade_command)
    if lines:
        console.print(Panel(Text("\n".join(lines)), border_style="cyan", padding=(1, 2), title="What's new"))
        console.print()
