"""记录上次运行的选择，并在下次运行时作为默认值提供。

交互流程每次都会询问相同问题，但只有部分问题可以通过环境变量跳过
（分析师选择没有对应环境变量）。保存的答案只用于预填提示，
用户按 Enter 即可接受；它不会跳过任何步骤，确保每次运行的选项都可见。

仅保存不同运行之间相对稳定的答案；ticker 和分析日期每次都可能变化，
记住它们可能不经意地建议过期日期。

读取时会按当前可选列表重新验证每个值，因为版本更新可能增加或淘汰
模型与提供商；已经下架的模型会被丢弃，而不是继续展示。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cli.models import AnalystType, AssetType
from cli.prompts import LISTED_LIVE, _llm_provider_table, filter_analysts_for_asset_type
from tradingagents.dataflows.files import replace_file
from tradingagents.llm_clients.model_catalog import get_model_options

_PREFS_PATH = Path(os.path.expanduser("~")) / ".tradingagents" / "cli_prefs.json"

REMEMBERED = (
    "output_language", "analysts", "research_depth", "llm_provider",
    "quick_think_llm", "deep_think_llm", "backend_url",
)


def load_last_run() -> dict:
    """读取上次运行的答案；没有可用数据时返回空字典。

    这些只是便捷配置：文件无法读取或已损坏时应视作没有默认值，
    不应因而中断用户操作。"""
    try:
        data = json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_last_run(selections: dict) -> None:
    """保存值得在下次运行时提供的答案；保存失败不应导致运行中断。"""
    kept = {k: v for k, v in selections.items() if k in REMEMBERED and v not in (None, "", [])}
    kept["analysts"] = [getattr(a, "value", a) for a in kept.get("analysts", [])] or None
    kept = {k: v for k, v in kept.items() if v is not None}
    try:
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 并发运行只会读取到替换前或替换后的完整文件。
        replace_file(_PREFS_PATH, lambda temp: Path(temp).write_text(json.dumps(kept, indent=2), encoding="utf-8"))
    except OSError:
        return


def sanitize(prefs: dict, asset_type) -> dict:
    """只保留当前版本仍可选择的历史答案。"""
    kept: dict = {}
    if isinstance(prefs.get("output_language"), str):
        kept["output_language"] = prefs["output_language"]
    if prefs.get("research_depth") in (1, 3, 5):
        kept["research_depth"] = prefs["research_depth"]

    known = {a.value for a in AnalystType}
    analysts = [a for a in prefs.get("analysts") or [] if a in known]
    allowed = filter_analysts_for_asset_type([AnalystType(a) for a in analysts], AssetType(asset_type))
    if allowed:
        kept["analysts"] = [a.value for a in allowed]

    provider = prefs.get("llm_provider")
    # 地区专属提供商（如 qwen-cn）在第二步提示中选择，
    # 因此此处应与主提供商菜单的基础 key 进行匹配。
    base = (provider or "").split("-cn")[0]
    if base and base in {key for _, key, _ in _llm_provider_table()}:
        kept["llm_provider"] = provider
        if isinstance(prefs.get("backend_url"), str) and prefs["backend_url"]:
            kept["backend_url"] = prefs["backend_url"]
        for field, mode in (("quick_think_llm", "quick"), ("deep_think_llm", "deep")):
            try:
                offered = {model for _, model in get_model_options(base, mode)}
            except KeyError:
                continue
            # 动态获取的模型不在静态目录中，因此保留任意模型 ID。
            listed_live = base in LISTED_LIVE and isinstance(prefs.get(field), str)
            if prefs.get(field) in offered or listed_live:
                kept[field] = prefs[field]
    return kept
