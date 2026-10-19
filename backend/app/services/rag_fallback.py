"""RAG 召回兜底：低召回重写 → 二次检索 → 仍低则拒答。

为什么要独立成模块
- 「兜底」是策略，不是存储；把它和 DB 解耦，才能用假的 search_fn 单测，
  也才能在换检索实现（纯向量 → 混合）时不动策略代码。

三段逻辑
1. 首轮检索后看**最高相关分**，低于阈值判为低召回。
2. 低召回时不是原样重试，而是**改写查询再检索一轮**：
   去掉地域限定（地域词常常把语义带偏）、只保留主干词、以及放宽到泛化词。
3. 二轮仍低于阈值 → **拒答**：返回空结果并标记 refused，
   让上层不要拿"没检索到"的上下文去生成报告（那等于编造）。

阈值默认 0.35，与线上检索配置一致；调用方可覆盖。
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# 首轮/二轮的召回分阈值
DEFAULT_MIN_SCORE = 0.35

# 地域词：命中这些词的查询在改写时优先剥离（地域限定最容易把语义带偏）
_REGION_WORDS = (
    "东南亚", "越南", "泰国", "印尼", "马来西亚", "中东", "欧美", "欧洲", "美国",
    "日本", "韩国", "非洲", "拉美", "俄罗斯", "印度",
)

# 泛化兜底查询：所有改写都失败时的最后一路
_FALLBACK_QUERY = "B2B工厂出海"

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


@dataclass
class RecallAssessment:
    """一次检索的可解释结论。"""

    best_score: float
    low_recall: bool
    reason: str


@dataclass
class FallbackResult:
    """兜底链路的结果。"""

    matches: list = field(default_factory=list)
    rounds: int = 0
    rewrites: list[str] = field(default_factory=list)
    best_score: float = 0.0
    refused: bool = False
    reason: str = ""


def assess_recall(matches: list, min_score: float = DEFAULT_MIN_SCORE) -> RecallAssessment:
    """按最高分判定召回质量。空结果直接判低召回。"""
    if not matches:
        return RecallAssessment(0.0, True, "no_match")

    best = max(float(getattr(m, "score", 0.0) or 0.0) for m in matches)
    if best < min_score:
        return RecallAssessment(best, True, "below_threshold")
    return RecallAssessment(best, False, "ok")


def rewrite_queries(query: str, known_facts: list[str] | None = None) -> list[str]:
    """生成改写查询（按"更可能召回"的顺序）。

    - r1：剥掉地域词后的查询（地域限定最常导致检索方向跑偏）
    - r2：只保留前两个中文词块（主干：行业 + 产品）
    - r3：泛化兜底词

    注意：这是**规则改写**，不是 LLM 重写。简历/面试里说"规则改写"是准确的。
    """
    query = (query or "").strip()
    out: list[str] = []

    stripped = query
    for w in _REGION_WORDS:
        stripped = stripped.replace(w, " ")
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped and stripped != query:
        out.append(stripped)

    runs = _CJK_RUN.findall(query)
    if len(runs) >= 2:
        out.append("".join(runs[:2]))
    elif runs:
        out.append(runs[0])

    for fact in (known_facts or []):
        fact = (fact or "").strip()
        if fact:
            out.append(fact)

    out.append(_FALLBACK_QUERY)

    # 去重保序
    seen: set[str] = set()
    deduped = []
    for q in out:
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


async def retrieve_with_fallback(
    search_fn: Callable[[str, int], Awaitable[list]],
    query: str,
    top_k: int = 10,
    min_score: float = DEFAULT_MIN_SCORE,
    max_rounds: int = 2,
    known_facts: list[str] | None = None,
) -> FallbackResult:
    """两轮检索 + 拒答判定。

    search_fn(query, top_k) -> list[匹配项]，匹配项需带 .score。
    """
    result = FallbackResult()

    first = await search_fn(query, top_k)
    result.rounds = 1
    assessment = assess_recall(first, min_score)
    result.best_score = assessment.best_score

    if not assessment.low_recall:
        result.matches = first
        result.reason = "round1_hit"
        return result

    if max_rounds < 2:
        result.matches = first
        result.refused = True
        result.reason = "round1_low_recall_no_retry"
        return result

    # ── 第二轮：改写后重检 ──
    best_matches = first
    best_score = assessment.best_score
    for rewritten in rewrite_queries(query, known_facts):
        result.rewrites.append(rewritten)
        matches = await search_fn(rewritten, top_k)
        result.rounds = 2
        a = assess_recall(matches, min_score)
        if a.best_score > best_score:
            best_score = a.best_score
            best_matches = matches
        if not a.low_recall:
            result.matches = matches
            result.best_score = a.best_score
            result.reason = f"round2_hit:{rewritten}"
            return result

    # ── 二轮仍低 → 拒答 ──
    result.matches = []          # 故意清空：不让上层拿低分片段去生成报告
    result.best_score = best_score
    result.refused = True
    result.reason = f"round2_still_low:{best_score:.2f}"
    return result
