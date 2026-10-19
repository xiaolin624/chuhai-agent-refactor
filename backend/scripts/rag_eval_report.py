"""RAG 检索离线评测：四档检索 × Recall@K / Hit@K / MRR@K + 延迟分位。

为什么要有这个脚本
- 简历里"混合检索把召回率从 A 提到 B""前 20 条 P95 为 X 秒"这类结论，
  必须能一条命令复现，否则面试一问就空。
- 这个脚本把「关键词 / 向量 / 混合 / 混合+精排」放在同一份评测集上跑，
  输出同一张表，差异只来自检索策略本身。

用法（在 backend 目录下，需配好 DATABASE_URL 与 Embedding Key）：
    python -m scripts.rag_eval_report
    python -m scripts.rag_eval_report --ks 3,20 --out reports/rag_eval

注意
- 向量侧调用 Embedding API；如果跑不通，脚本会明确标 `SKIPPED（需要 Embedding API）`，
  **不会用假数据补位**。
- 延迟是「单次检索墙钟时间」，含一次 Embedding 调用；因此 keyword 档会明显快于其余三档。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import async_session  # noqa: E402
from app.services import rag_retriever as R  # noqa: E402
from app.services.rag_eval import EVAL_QUERIES  # noqa: E402


def percentile(values: list[float], p: float) -> float | None:
    """nearest-rank 百分位。不插值，避免用 2 个样本编出一个 P99。"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = max(1, min(len(s), math.ceil(p / 100.0 * len(s))))
    return s[idx - 1]


def recall_at(relevant: set[str], returned: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(returned[:k])) / len(relevant)


def hit_at(relevant: set[str], returned: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if relevant & set(returned[:k]) else 0.0


def mrr_at(relevant: set[str], returned: list[str], k: int) -> float:
    for i, title in enumerate(returned[:k], start=1):
        if title in relevant:
            return 1.0 / i
    return 0.0


async def run_mode(db, mode: str, ks: list[int], top_k: int) -> dict:
    per_query: list[dict] = []
    latencies: list[float] = []
    error: str | None = None

    for case in EVAL_QUERIES:
        query = case["query"]
        relevant = set(case["relevant_titles"])
        started = time.perf_counter()
        try:
            matches = await R.hybrid_search(db, query, top_k=top_k, mode=mode)
        except Exception as exc:  # 向量侧缺 Key / 连不上时，整档标记跳过
            error = f"{type(exc).__name__}: {exc}"
            break
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        returned = [m.title for m in matches]
        per_query.append({
            "query": query,
            "returned": returned,
            "relevant": sorted(relevant),
            "recall": {k: recall_at(relevant, returned, k) for k in ks},
            "hit": {k: hit_at(relevant, returned, k) for k in ks},
            "mrr": {k: mrr_at(relevant, returned, k) for k in ks},
            "latency_ms": round(elapsed_ms, 1),
        })

    if error:
        return {"mode": mode, "status": "SKIPPED", "error": error, "n": 0}

    n = len(per_query) or 1
    return {
        "mode": mode,
        "status": "OK",
        "n": len(per_query),
        "recall": {k: sum(q["recall"][k] for q in per_query) / n for k in ks},
        "hit": {k: sum(q["hit"][k] for q in per_query) / n for k in ks},
        "mrr": {k: sum(q["mrr"][k] for q in per_query) / n for k in ks},
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "n": len(latencies),
        },
        "per_query": per_query,
    }


def to_markdown(results: list[dict], ks: list[int], top_k: int) -> str:
    lines = [
        "# RAG 检索离线评测",
        "",
        f"- 评测集：{len(EVAL_QUERIES)} 条 query",
        f"- 返回条数 top_k：{top_k}",
        f"- 指标：Recall@K / Hit@K / MRR@K，K = {', '.join(map(str, ks))}",
        f"- 延迟：单次检索墙钟时间（含一次 Embedding 调用），nearest-rank 分位",
        "",
        "| 检索档位 | 样本 | " + " | ".join(f"Recall@{k}" for k in ks) + " | "
        + " | ".join(f"Hit@{k}" for k in ks) + " | MRR | P50(ms) | P95(ms) |",
        "|---" * (5 + len(ks) * 2) + "|",
    ]
    for r in results:
        if r["status"] != "OK":
            lines.append(f"| {r['mode']} | SKIPPED | " + " | ".join(["—"] * (len(ks) * 2 + 3)) + " |")
            continue
        cells = [r["mode"], str(r["n"])]
        cells += [f"{r['recall'][k]:.1%}" for k in ks]
        cells += [f"{r['hit'][k]:.1%}" for k in ks]
        cells += [
            f"{r['mrr'][ks[-1]]:.4f}",
            f"{r['latency_ms']['p50']}",
            f"{r['latency_ms']['p95']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    skipped = [r for r in results if r["status"] != "OK"]
    if skipped:
        lines += ["", "## 跳过的档位", ""]
        for r in skipped:
            lines.append(f"- `{r['mode']}`：{r['error']}")

    lines += [
        "",
        "## 口径声明",
        "",
        "- 关键词侧是自实现 Okapi BM25（中文二元组分词），不是 Elasticsearch。",
        "- 融合用 RRF（k=60），不是加权求和。",
        "- 「精排」是词面重排（BM25 归一 + 标题词项覆盖），**不是模型 Rerank**。",
        "- 语料只有十几篇文档，Recall 绝对值会偏高；这份表用于**同语料下的横向对比**，",
        "  不能当成生产环境的召回率。",
    ]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", default="3,20")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--modes", default=",".join(R.ALL_MODES))
    parser.add_argument("--out", default="reports/rag_eval")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    results = []
    async with async_session() as db:
        for mode in modes:
            print(f"[run] {mode} ...", flush=True)
            results.append(await run_mode(db, mode, ks, args.top_k))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "eval_size": len(EVAL_QUERIES),
        "top_k": args.top_k,
        "ks": ks,
        "results": [{k: v for k, v in r.items() if k != "per_query"} for r in results],
        "per_query": {r["mode"]: r.get("per_query", []) for r in results},
    }
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(results, ks, args.top_k), encoding="utf-8")

    print()
    print(to_markdown(results, ks, args.top_k))
    print()
    print(f"JSON -> {json_path}")
    print(f"MD   -> {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
