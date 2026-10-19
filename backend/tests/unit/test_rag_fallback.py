"""召回兜底的离线单测：用假的 search_fn，不碰 DB、不碰 API。"""

from __future__ import annotations

import pytest

from app.services.rag_fallback import (
    assess_recall,
    retrieve_with_fallback,
    rewrite_queries,
)


class FakeMatch:
    def __init__(self, title: str, score: float) -> None:
        self.title = title
        self.score = score


def test_assess_recall_empty_is_low():
    a = assess_recall([])
    assert a.low_recall is True
    assert a.reason == "no_match"


def test_assess_recall_threshold():
    low = assess_recall([FakeMatch("x", 0.2)], min_score=0.35)
    assert low.low_recall is True and low.reason == "below_threshold"

    ok = assess_recall([FakeMatch("x", 0.2), FakeMatch("y", 0.8)], min_score=0.35)
    assert ok.low_recall is False
    assert ok.best_score == 0.8


def test_reward_strips_region_words():
    out = rewrite_queries("越南建厂需要注意什么")
    assert any("越南" not in q for q in out[:1])
    assert out[-1] == "B2B工厂出海"


def test_rewrite_always_has_fallback_and_dedupes():
    out = rewrite_queries("B2B工厂出海")
    assert out[-1] == "B2B工厂出海"
    assert len(out) == len(set(out))


def test_round1_hit_skips_rewrite():
    calls: list[str] = []

    async def search_fn(q: str, k: int):
        calls.append(q)
        return [FakeMatch("doc", 0.9)]

    import asyncio
    res = asyncio.run(retrieve_with_fallback(search_fn, "东南亚物流", top_k=5))
    assert res.rounds == 1
    assert res.refused is False
    assert res.reason == "round1_hit"
    assert len(calls) == 1


def test_round2_rescues_low_recall():
    async def search_fn(q: str, k: int):
        # 只有剥掉地域词的查询能召回
        if "东南亚" not in q:
            return [FakeMatch("doc", 0.8)]
        return [FakeMatch("doc", 0.1)]

    import asyncio
    res = asyncio.run(retrieve_with_fallback(search_fn, "东南亚 跨境物流", top_k=5))
    assert res.rounds == 2
    assert res.refused is False
    assert res.reason.startswith("round2_hit")
    assert res.matches


def test_round2_still_low_refuses_and_clears():
    async def search_fn(q: str, k: int):
        return [FakeMatch("doc", 0.05)]

    import asyncio
    res = asyncio.run(retrieve_with_fallback(search_fn, "完全无关的问题", top_k=5))
    assert res.refused is True
    assert res.matches == []          # 关键：不能把低分片段交出去
    assert res.reason.startswith("round2_still_low")


def test_max_rounds_one_refuses_without_retry():
    calls: list[str] = []

    async def search_fn(q: str, k: int):
        calls.append(q)
        return [FakeMatch("doc", 0.05)]

    import asyncio
    res = asyncio.run(retrieve_with_fallback(search_fn, "问题", max_rounds=1))
    assert res.refused is True
    assert res.rounds == 1
    assert len(calls) == 1
