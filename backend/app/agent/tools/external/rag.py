"""RAG 检索工具：混合检索 + 低召回兜底 + 延迟观测。

相对旧版的三处改动
1. 检索从「纯向量」换成「混合检索 + 词面精排」；Embedding 不可用时**降级为 BM25**，
   而不是直接返回空（旧的 `except: return []` 会把故障伪装成"没检索到"）。
2. 接入两轮兜底：低召回 → 规则改写重检 → 仍低则 `refused=True`，
   让上层知道"知识库没有"，不要拿空上下文生成报告。
3. 输出带上 `elapsed_ms / rounds / mode_used / best_score`，延迟可被统计。

口径提醒：`rewrites` 是**规则改写**（剥地域词、取主干、泛化兜底），
不是 LLM 重写；对外表述不要说成"用模型改写查询"。
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from app.agent.tools.base import ToolContext, ToolResult
from app.db.session import async_session
from app.schemas.rag import RagDocumentMatch
from app.services import rag_retriever as R
from app.services.rag_fallback import DEFAULT_MIN_SCORE, retrieve_with_fallback

logger = logging.getLogger(__name__)


class RagSearchInput(BaseModel):
    query: str
    top_k: int = 10
    # 低于该分判为低召回，进入改写重检
    min_score: float = DEFAULT_MIN_SCORE
    # 1 = 不重试；2 = 首轮 + 改写轮
    max_rounds: int = 2
    # keyword | vector | hybrid | hybrid_rerank
    mode: str = R.MODE_HYBRID_RERANK


class RagSearchOutput(BaseModel):
    matches: list[RagDocumentMatch] = Field(default_factory=list)
    # 以下为新增的可观测字段，带默认值，旧调用方不受影响
    refused: bool = False
    rounds: int = 1
    best_score: float = 0.0
    mode_used: str = ""
    rewrites: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    reason: str = ""


async def _search_with_degrade(db, query: str, top_k: int, mode: str) -> tuple[list, str]:
    """先按请求的模式检索；Embedding 不可用时降级为纯 BM25。

    降级是显式的：返回值里带实际使用的模式，便于日志与评测区分。
    """
    if mode == R.MODE_KEYWORD:
        return await R.hybrid_search(db, query, top_k=top_k, mode=R.MODE_KEYWORD), R.MODE_KEYWORD

    try:
        return await R.hybrid_search(db, query, top_k=top_k, mode=mode), mode
    except Exception as exc:  # 缺 Key / 网络 / 向量库不可用
        logger.warning("rag hybrid search degraded to keyword: %s: %s", type(exc).__name__, exc)
        matches = await R.hybrid_search(db, query, top_k=top_k, mode=R.MODE_KEYWORD)
        return matches, R.MODE_KEYWORD + "(degraded)"


async def rag_search_handler(
    inp: RagSearchInput,
    ctx: ToolContext,
) -> ToolResult:
    started = time.perf_counter()
    mode_used = inp.mode

    try:
        async with async_session() as db:

            async def search_fn(query: str, top_k: int) -> list:
                nonlocal mode_used
                matches, used = await _search_with_degrade(db, query, top_k, inp.mode)
                mode_used = used
                return matches

            result = await retrieve_with_fallback(
                search_fn,
                inp.query,
                top_k=inp.top_k,
                min_score=inp.min_score,
                max_rounds=inp.max_rounds,
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "rag.search query=%r mode=%s rounds=%d best=%.3f refused=%s elapsed_ms=%.1f",
            inp.query, mode_used, result.rounds, result.best_score, result.refused, elapsed_ms,
        )

        return ToolResult(data=RagSearchOutput(
            matches=result.matches,
            refused=result.refused,
            rounds=result.rounds,
            best_score=round(result.best_score, 4),
            mode_used=mode_used,
            rewrites=result.rewrites,
            elapsed_ms=elapsed_ms,
            reason=result.reason,
        ))
    except Exception as exc:
        logger.exception("rag.search failed: %s", exc)
        return ToolResult(data=RagSearchOutput(
            matches=[],
            refused=True,
            rounds=0,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            reason=f"error:{type(exc).__name__}",
        ))
