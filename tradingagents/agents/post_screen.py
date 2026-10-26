"""在设置 ``TYPESAFE_API_KEY`` 时，使用 TypeSafe 的 Jev 筛选社交帖子。

Jev 为类型化问题返回经过校准的概率。每条 StockTwits 或 Reddit
帖子会被询问两个问题：是否与目标标的有关，以及对该股票偏多还是偏空。
明确无关的帖子被过滤，其余帖子按立场聚合后提供给情绪分析师。

使用 TypeSafe SDK 的 ``TYPESAFE_API_KEY``、``TYPESAFE_DEFAULT_MODEL``
和 ``TYPESAFE_BASE_URL`` 配置（OpenRouter 的 Jev 接口位于
``https://openrouter.ai/api``）。没有密钥时不执行筛选；
若请求失败，保留未筛选帖子，并说明筛选服务不可用。"""

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from tradingagents.agents.context import resolve_instrument_identity

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_MODEL = "jev-latest"
_RETRY_STATUSES = (429, 529)    # rate limited, overloaded: back off and retry
_TRANSIENT = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)
_ATTEMPTS = 3
_MAX_WAIT = 30.0
_TIMEOUT = 15.0
_WORKERS = 16                   # well inside the documented 1,200 requests per minute

# 只有明确判定与目标无关时才丢弃帖子，无法确定的中间情况保留。
_OFF_TOPIC_BELOW = 0.3
# 只有 Jev 对立场有足够把握时才将其计入统计。
_STANCE_CONFIDENCE = 0.5

QUESTIONS = {
    "about": {
        "type": "noul",
        "instructions": "Is `post` about `instrument`: the company, its stock, its products or its outlook?",
        "criteria": {
            "true": "`post` discusses `instrument` itself.",
            "false": "`post` names `instrument` only in passing or in a list of tickers, is spam or "
                     "promotion, or is about a different company.",
        },
    },
    "stance": {
        "type": "choice",
        "instructions": "What does the author of `post` expect for the stock price of `instrument`?",
        "criteria": {
            "bullish": "The author expects `instrument`'s stock to rise, or is buying or holding it long.",
            "bearish": "The author expects `instrument`'s stock to fall, or is selling or shorting it.",
            "neutral": "The author gives no view of their own on `instrument`'s stock: a question, "
                       "news without opinion, or someone else's view quoted.",
        },
    },
}


class TypeSafeError(Exception):
    """表示 System One 请求未能返回有效答案的异常。"""


def system_one(state, questions: dict) -> dict[str, dict]:
    """针对 ``state`` 询问 ``questions``，以问题名作为键返回答案。

    限流、过载或连接中断时执行退避重试，并尊重 ``Retry-After``；
    其他异常立即抛出 ``TypeSafeError``。"""
    body = {
        "state": state,
        "model": os.environ.get("TYPESAFE_DEFAULT_MODEL") or _DEFAULT_MODEL,
        "questions": questions,
    }
    headers = {"Authorization": f"Bearer {os.environ.get('TYPESAFE_API_KEY', '')}"}
    backoff, retry_after = 1.0, None
    for attempt in range(_ATTEMPTS):
        if attempt:
            time.sleep(retry_after if retry_after is not None else backoff * random.uniform(0.8, 1.2))
            backoff *= 2
        try:
            response = requests.post(_url(), json=body, headers=headers, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            failure, retry_after = type(exc).__name__, None
            if isinstance(exc, _TRANSIENT):
                continue
            break
        if response.status_code == 200:
            return _answers(response, questions)
        failure, retry_after = f"HTTP {response.status_code}", _retry_after(response)
        if response.status_code not in _RETRY_STATUSES:
            break
    raise TypeSafeError(failure)


def _retry_after(response) -> float | None:
    try:
        return min(max(0.0, float(response.headers.get("retry-after"))), _MAX_WAIT)
    except (TypeError, ValueError):
        return None


def _answers(response, questions: dict) -> dict[str, dict]:
    try:
        payload = response.json()
        answers = payload["answers"]
        if all(answers[q]["type"] == spec["type"] for q, spec in questions.items()):
            logger.debug("TypeSafe answered with %s", payload.get("model"))
            return answers
    except (ValueError, KeyError, TypeError):
        pass
    raise TypeSafeError("malformed response")


def _stance(answer: dict) -> str:
    choice = answer["choice"]
    if choice not in QUESTIONS["stance"]["criteria"] or answer["confidence"] < _STANCE_CONFIDENCE:
        return "unclear"
    return choice


def _url() -> str:
    """System One 接口地址：在 TypeSafe SDK 使用的基础地址后添加 /v1/systemone。"""
    base = os.environ.get("TYPESAFE_BASE_URL") or _DEFAULT_BASE_URL
    return base.rstrip("/") + "/v1/systemone"


def jev_screen(ticker: str):
    """创建用于社交数据获取器的帖子筛选器；没有 TypeSafe 密钥时返回 None。

    筛选器接收帖子文本，为每条帖子返回保留标志，
    并生成一行放在数据区块开头的说明。"""
    if not os.environ.get("TYPESAFE_API_KEY"):
        return None
    name = resolve_instrument_identity(ticker).get("company_name")
    instrument = f"{name} ({ticker})" if name else ticker

    def screen(posts: list[str]) -> tuple[list[bool], str]:
        try:
            answers = _ask_each(instrument, posts)
            keep = [a["about"]["noul"] >= _OFF_TOPIC_BELOW for a in answers]
            stances = [_stance(a["stance"]) for a, kept in zip(answers, keep, strict=True) if kept]
        except (KeyError, TypeError):
            return _unscreened(instrument, posts, "malformed response")
        except TypeSafeError as exc:
            return _unscreened(instrument, posts, str(exc))
        counts = ", ".join(f"{stances.count(s)} {s}" for s in ("bullish", "bearish", "neutral", "unclear"))
        return keep, (
            f"Screened by Jev: {len(stances)} of the {len(posts)} posts fetched are about "
            f"{instrument}; their stance on its stock: {counts}."
        )

    return screen


def _unscreened(instrument: str, posts: list[str], failure: str) -> tuple[list[bool], str]:
    logger.warning("Jev screening failed for %s: %s", instrument, failure)
    return [True] * len(posts), f"<Jev screening unavailable ({failure}); posts are unscreened>"


def _ask_each(instrument: str, posts: list[str]) -> list[dict]:
    """每个帖子发起一次请求。首次失败立即抛异常；
    尚未发出的请求被取消，已在执行的请求完成后其结果不再读取。"""
    answers: list = [None] * len(posts)
    pool = ThreadPoolExecutor(max_workers=_WORKERS)
    try:
        futures = {
            pool.submit(system_one, {"instrument": instrument, "post": post}, QUESTIONS): i
            for i, post in enumerate(posts)
        }
        for future in as_completed(futures):
            answers[futures[future]] = future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return answers
