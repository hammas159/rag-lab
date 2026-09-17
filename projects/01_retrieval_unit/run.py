"""01 - What should a retrieval unit be?

LongRAG argues that chunking is a self-inflicted wound: retrieve whole documents
and let the reader find the answer inside. The opposite intuition -- retrieve
sentences, because a sentence is precise -- is just as widely repeated.

Both cannot be right, and the trade is measurable without a language model.
Smaller units mean a sharper match and less irrelevant text; larger units mean
the evidence is less likely to be split across two units, neither of which
retrieves on its own.

Three granularities over the same corpus, the same queries, the same retrievers:

    title_only  the article title alone          (~3 tokens)
    sentence    every sentence its own unit      (~25 tokens, with title prefix)
    paragraph   the HotpotQA unit                (~92 tokens, with title prefix)

HotpotQA gives one paragraph per title, so "whole document" and "paragraph" are
the same thing here -- there is nothing larger to retrieve. `title_only` takes
the comparison in the other direction instead, to the smallest unit that exists.

A unit is credited as retrieving a gold document if it belongs to that document,
so the metric compares like with like: the question is always "did the gold
paragraph's content reach the top k", not "did we return more text".

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
from shared.retrieval import BM25, Dense, rrf  # noqa: E402

HERE = Path(__file__).resolve().parent
N_QUESTIONS = 1500
DEPTH = 20
POOL = 50


def build_units(corpus, granularity: str) -> tuple[list[str], list[int]]:
    """Return (unit_texts, owning_doc_id per unit)."""
    texts: list[str] = []
    owner: list[int] = []
    for p in corpus.passages:
        if granularity == "sentence":
            for s in p.sentences:
                s = s.strip()
                if s:
                    texts.append(f"{p.title}. {s}")
                    owner.append(p.doc_id)
        elif granularity == "paragraph":
            texts.append(f"{p.title}. {p.text}")
            owner.append(p.doc_id)
        elif granularity == "title_only":
            texts.append(p.title)
            owner.append(p.doc_id)
        else:
            raise ValueError(granularity)
    return texts, owner


def to_docs(unit_ranking: list[int], owner: list[int], k: int) -> list[int]:
    """Collapse a ranking of units into a ranking of documents, first hit wins."""
    seen: list[int] = []
    for unit_id in unit_ranking:
        doc_id = owner[unit_id]
        if doc_id not in seen:
            seen.append(doc_id)
            if len(seen) >= k:
                break
    return seen


def main() -> None:
    corpus, queries = load(N_QUESTIONS)
    print(f"corpus: {corpus.summary()}")
    print(f"queries: {len(queries)}\n")

    results = {}
    for granularity in ("title_only", "sentence", "paragraph"):
        texts, owner = build_units(corpus, granularity)
        unit_tokens = sum(len(t.split()) for t in texts) / max(len(texts), 1)
        print(f"=== {granularity}: {len(texts)} units, {unit_tokens:.0f} tokens each (mean)")

        t0 = time.time()
        bm25 = BM25().index(texts)
        dense = Dense().index(texts)
        index_s = time.time() - t0

        row = {
            "units": len(texts),
            "mean_unit_tokens": round(unit_tokens, 1),
            "index_seconds": round(index_s, 1),
        }

        for name in ("bm25", "dense", "hybrid"):
            t0 = time.time()
            rankings = []
            for q in queries:
                if name == "bm25":
                    units = [u for u, _ in bm25.search(q.question, POOL)]
                elif name == "dense":
                    units = [u for u, _ in dense.search(q.question, POOL)]
                else:
                    fused = rrf(
                        [bm25.search(q.question, POOL), dense.search(q.question, POOL)], k=POOL
                    )
                    units = [u for u, _ in fused]
                rankings.append((to_docs(units, owner, DEPTH), q.gold_doc_ids))
            scores = evaluate(rankings, seconds=time.time() - t0)
            row[name] = scores.as_dict()
            d = row[name]
            print(
                f"  {name:8} mrr={d['mrr']:.3f}  recall@10={d['recall'][10]:.3f}  "
                f"answerable@10={d['answerable'][10]:.3f}  ({d['seconds']}s)"
            )
        results[granularity] = row
        print()

    out = HERE / "results.json"
    out.write_text(
        json.dumps({"n_questions": len(queries), "depth": DEPTH, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"written to {out.name}")


if __name__ == "__main__":
    main()
