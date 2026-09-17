"""08 - Speculative retrieval: can a cheap drafter plus a verifier beat an expensive retriever?

Speculative decoding runs a small model to draft and a large one to check, and
wins because checking is cheaper than generating. Speculative RAG borrows the
shape: a cheap retriever proposes, an expensive one verifies.

Retrieval has exactly the right asymmetry for this. BM25 scores the whole corpus
in milliseconds and ranks badly. A cross-encoder ranks well and costs a forward
pass **per candidate**, so it can never look at the whole corpus -- it can only
reorder what something else proposed.

So the real question is not whether reranking helps. It is **how weak the drafter
is allowed to be**. If BM25 alone can hand the cross-encoder a good enough pool,
the dense index -- which costs 47 seconds to build and 9 GB of memory at scale --
buys nothing that the reranker does not recover.

    drafter    BM25 (3 ms/query)  or  dense  or  hybrid
    verifier   cross-encoder/ms-marco-MiniLM-L-6-v2 over the drafted pool
    swept      pool size, because that is the entire cost knob

    python run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.corpus import load  # noqa: E402
from shared.metrics import evaluate  # noqa: E402
from shared.pipeline import Index  # noqa: E402
from shared.retrieval import Reranker, rrf  # noqa: E402

HERE = Path(__file__).resolve().parent
N_QUESTIONS = 500  # the cross-encoder is the cost here: pool x queries forward passes
DEPTH = 20


def main() -> None:
    corpus, queries = load(1500)
    queries = queries[:N_QUESTIONS]
    index = Index.build(corpus)
    reranker = Reranker()
    texts = index.texts
    print(f"corpus: {len(corpus)} passages, queries: {len(queries)}\n")

    def draft_bm25(q: str, pool: int) -> list[int]:
        return [d for d, _ in index.bm25.search(q, pool)]

    def draft_dense(q: str, pool: int) -> list[int]:
        return [d for d, _ in index.dense.search(q, pool)]  # type: ignore[union-attr]

    def draft_hybrid(q: str, pool: int) -> list[int]:
        return [
            d
            for d, _ in rrf(
                [index.bm25.search(q, pool), index.dense.search(q, pool)],  # type: ignore[union-attr]
                k=pool,
            )
        ]

    drafters = {"bm25": draft_bm25, "dense": draft_dense, "hybrid": draft_hybrid}

    results = {"no_verifier": {}, "with_verifier": {}}

    print("drafter alone (no verifier)")
    for name, draft in drafters.items():
        t0 = time.time()
        ranked = [(draft(q.question, DEPTH), q.gold_doc_ids) for q in queries]
        d = evaluate(ranked, seconds=time.time() - t0).as_dict()
        results["no_verifier"][name] = d
        print(f"  {name:8} answerable@10={d['answerable'][10]:.3f}  ({d['seconds']}s)")

    print("\ndrafter + cross-encoder verifier")
    for name, draft in drafters.items():
        for pool in (20, 50, 100):
            t0 = time.time()
            ranked = []
            for q in queries:
                candidates = draft(q.question, pool)
                pairs = [(d, texts[d]) for d in candidates]
                order = [d for d, _ in reranker.rerank(q.question, pairs, k=DEPTH)]
                ranked.append((order, q.gold_doc_ids))
            d = evaluate(ranked, seconds=time.time() - t0).as_dict()
            d["pool"] = pool
            results["with_verifier"][f"{name}@{pool}"] = d
            print(
                f"  {name:8} pool={pool:<4} answerable@10={d['answerable'][10]:.3f}  "
                f"({d['seconds']}s, {d['seconds'] / len(queries) * 1000:.0f} ms/query)"
            )

    out = HERE / "results.json"
    out.write_text(
        json.dumps({"n_questions": len(queries), "depth": DEPTH, **results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()
