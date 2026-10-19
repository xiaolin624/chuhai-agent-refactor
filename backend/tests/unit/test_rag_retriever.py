"""混合检索的离线单测：不碰数据库、不碰 Embedding API。

覆盖：
- 中英混排分词
- BM25 打分的基本性质（命中比不命中高、标题加权生效）
- RRF 融合的排序语义（两路都靠前的排第一）
- 词面精排能把标题命中的候选提上来
"""

from __future__ import annotations

from app.services.rag_retriever import (
    BM25Index,
    Doc,
    lexical_rerank,
    rrf_fuse,
    tokenize,
)


DOCS = [
    Doc("东南亚市场B2B工厂出海要点", "越南泰国印尼关税优势，物流成本低，建议从阿里国际站切入。"),
    Doc("TikTok短视频B2B获客策略", "工厂车间拍摄能展示制造能力，每周发布3-5条，搭配WhatsApp承接询盘。"),
    Doc("产品出口认证与合规基础", "欧盟需要CE认证，美国需要FDA或FCC，认证周期2-6个月。"),
]


def test_tokenize_chinese_bigram():
    tokens = tokenize("海外询盘")
    assert "海外" in tokens
    assert "外询" in tokens
    assert "询盘" in tokens


def test_tokenize_latin_and_digits():
    tokens = tokenize("TikTok 3-5 条")
    assert "tiktok" in tokens
    # 连字符数字整体保留成一个词项（避免把「3-5」这类量词范围切断）
    assert "3-5" in tokens
    # 中文单字也进词表（「条」只有一个字，二元组切不出来）
    assert "条" in tokens


def test_bm25_scores_positive_only_for_hits():
    idx = BM25Index(DOCS)
    scores = idx.scores("CE认证")
    assert "产品出口认证与合规基础" in scores
    # 完全不相关的文档不应该出现
    assert "TikTok短视频B2B获客策略" not in scores


def test_bm25_title_boost_matters():
    idx = BM25Index(DOCS)
    # 「出海」只出现在第一篇标题里
    scores = idx.scores("出海")
    assert set(scores) == {"东南亚市场B2B工厂出海要点"}


def test_rrf_prefers_consensus():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    # a 和 b 都在两路里，且都在前二；c/d 只出现一次
    assert set(fused[:2]) == {"a", "b"}
    assert set(fused[2:]) == {"c", "d"}


def test_rrf_handles_empty():
    assert rrf_fuse([[], []]) == []


def test_lexical_rerank_promotes_title_match():
    idx = BM25Index(DOCS)
    candidates = ["产品出口认证与合规基础", "TikTok短视频B2B获客策略"]
    reranked = lexical_rerank("TikTok获客", candidates, idx)
    assert reranked[0] == "TikTok短视频B2B获客策略"


def test_lexical_rerank_empty():
    idx = BM25Index(DOCS)
    assert lexical_rerank("任意", [], idx) == []
