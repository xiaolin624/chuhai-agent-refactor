"""混合检索：BM25 关键词侧 + 向量语义侧 + RRF 融合 + 词面精排。

为什么单独一个模块，而不是改 `rag_repository.search_rag_context`
- 原函数是纯向量检索，被工具链直接调用；新增能力做成独立模块，**零侵入**。
- 评测要能分别跑「关键词 / 向量 / 混合 / 混合+精排」四档做对照，
  所以把「排序」和「取文档」拆开，排序结果统一用 title 列表表示。

能力边界（不要在简历/面试里说过头）
- 关键词侧是**自实现的 Okapi BM25**，不是 Elasticsearch。
- 精排是**词面重排**（标题命中 + 词项覆盖 + BM25 归一），
  **不是** cross-encoder / Rerank 模型。说「词面精排」是准确的，
  说「接了 Rerank 模型」是假的。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag_document import RagDocument
from app.schemas.rag import RagDocumentMatch
from app.services.deepseek_client import DeepSeekClient

# 检索档位
MODE_KEYWORD = "keyword"          # 纯 BM25
MODE_VECTOR = "vector"            # 纯向量（pgvector 余弦）
MODE_HYBRID = "hybrid"            # RRF 融合
MODE_HYBRID_RERANK = "hybrid_rerank"  # RRF + 词面精排

ALL_MODES = (MODE_KEYWORD, MODE_VECTOR, MODE_HYBRID, MODE_HYBRID_RERANK)

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+\-\.]*")

# BM25 参数（Okapi 常用默认值）
_K1 = 1.5
_B = 0.75
# RRF 平滑常数（原论文取 60）
_RRF_K = 60
# 标题在文档词频里的重复次数（标题命中权重）
_TITLE_BOOST = 2


def tokenize(text: str) -> list[str]:
    """中英混排分词：中文按字符二元组，英文数字按词。

    不引 jieba 是刻意的：知识库规模小（十几篇），二元组召回足够，
    且避免为一个检索 demo 引入一个 40MB 的分词依赖。
    """
    text = (text or "").lower()
    tokens: list[str] = []

    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))

    tokens.extend(_LATIN.findall(text))
    return tokens


@dataclass
class Doc:
    title: str
    content: str
    source: str | None = None


class BM25Index:
    """Okapi BM25，语料在内存里（十几篇文档，够用）。"""

    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        df: Counter[str] = Counter()

        for d in docs:
            # 标题重复 _TITLE_BOOST 次，让标题命中天然占优
            tokens = tokenize(d.title) * _TITLE_BOOST + tokenize(d.content)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._len.append(len(tokens))
            df.update(tf.keys())

        n = len(docs) or 1
        self._avgdl = (sum(self._len) / n) if self._len else 0.0
        # BM25 的 IDF（加 1 防负值）
        self._idf = {
            t: math.log(1 + (n - c + 0.5) / (c + 0.5))
            for t, c in df.items()
        }

    def scores(self, query: str) -> dict[str, float]:
        """返回 title -> 分数。"""
        q_tokens = tokenize(query)
        out: dict[str, float] = {}
        if not q_tokens or not self.docs:
            return out

        for doc, tf, dl in zip(self.docs, self._tf, self._len):
            score = 0.0
            norm = _K1 * (1 - _B + _B * (dl / self._avgdl if self._avgdl else 1))
            for t in q_tokens:
                f = tf.get(t)
                if not f:
                    continue
                score += self._idf.get(t, 0.0) * (f * (_K1 + 1)) / (f + norm)
            if score > 0:
                out[doc.title] = score
        return out


def rrf_fuse(rankings: list[list[str]], k: int = _RRF_K) -> list[str]:
    """Reciprocal Rank Fusion：把多路有序列表融合成一个。

    不需要两路分数量纲一致，这是选它而不是加权求和的原因。
    """
    agg: dict[str, float] = {}
    for ranking in rankings:
        for rank, title in enumerate(ranking, start=1):
            agg[title] = agg.get(title, 0.0) + 1.0 / (k + rank)
    return [t for t, _ in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))]


def lexical_rerank(query: str, candidates: list[str], bm25: BM25Index) -> list[str]:
    """词面精排：BM25 归一 + 标题词项覆盖率。

    这是「重排」的一种，但不是模型重排：没有 cross-encoder，没有训练。
    """
    if not candidates:
        return []

    raw = bm25.scores(query)
    top = max((raw.get(t, 0.0) for t in candidates), default=0.0) or 1.0
    q_tokens = set(tokenize(query))
    by_title = {d.title: d for d in bm25.docs}

    def score(title: str) -> float:
        doc = by_title.get(title)
        title_tokens = set(tokenize(doc.title)) if doc else set()
        coverage = (len(q_tokens & title_tokens) / len(q_tokens)) if q_tokens else 0.0
        return 0.6 * (raw.get(title, 0.0) / top) + 0.4 * coverage

    return sorted(candidates, key=lambda t: (-score(t), t))


async def _load_docs(db: AsyncSession) -> list[Doc]:
    stmt = select(RagDocument.title, RagDocument.content, RagDocument.source)
    rows = (await db.execute(stmt)).all()
    return [Doc(title=r.title, content=r.content, source=r.source) for r in rows]


async def _vector_order(db: AsyncSession, query: str, candidates: int) -> list[str]:
    """pgvector 余弦距离排序，返回 title 列表。"""
    client = DeepSeekClient()
    embedding = await client.embed_text(query)
    stmt = (
        select(RagDocument.title)
        .where(RagDocument.embedding.isnot(None))
        .order_by(RagDocument.embedding.cosine_distance(embedding))
        .limit(candidates)
    )
    rows = (await db.execute(stmt)).all()
    return [r.title for r in rows]


async def hybrid_search(
    db: AsyncSession,
    query: str,
    top_k: int = 10,
    mode: str = MODE_HYBRID_RERANK,
    candidates: int = 30,
) -> list[RagDocumentMatch]:
    """四档检索的统一入口。

    mode:
      keyword        纯 BM25
      vector         纯向量
      hybrid         RRF 融合
      hybrid_rerank  RRF 融合 + 词面精排（默认）
    """
    docs = await _load_docs(db)
    if not docs:
        return []

    bm25 = BM25Index(docs)
    by_title = {d.title: d for d in docs}

    keyword_order = [
        t for t, _ in sorted(bm25.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
    ][:candidates]

    if mode == MODE_KEYWORD:
        order = keyword_order
    elif mode == MODE_VECTOR:
        order = await _vector_order(db, query, candidates)
    elif mode == MODE_HYBRID:
        order = rrf_fuse([await _vector_order(db, query, candidates), keyword_order])
    elif mode == MODE_HYBRID_RERANK:
        fused = rrf_fuse([await _vector_order(db, query, candidates), keyword_order])
        order = lexical_rerank(query, fused[:candidates], bm25)
    else:
        raise ValueError(f"unknown mode: {mode}")

    matches: list[RagDocumentMatch] = []
    for rank, title in enumerate(order[:top_k], start=1):
        doc = by_title.get(title)
        if doc is None:
            continue
        matches.append(
            RagDocumentMatch(
                title=doc.title,
                content=doc.content,
                source=doc.source,
                score=round(1.0 / rank, 4),
            )
        )
    return matches
