"""The baselines every variant in this lab is measured against.

Each strategy is a function from a question to a ranked list of doc_ids. Keeping
that interface narrow is what lets a variant be dropped in as one more row
without touching the harness or the metrics.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .corpus import Corpus, Query
from .metrics import Scores, evaluate
from .retrieval import BM25, Dense, Reranker, rrf

Ranker = Callable[[str, int], list[int]]


@dataclass
class Index:
    """Built once, shared by every strategy, so nothing is measured twice."""

    corpus: Corpus
    bm25: BM25
    dense: Dense | None = None
    _texts: list[str] | None = None

    @property
    def texts(self) -> list[str]:
        if self._texts is None:
            self._texts = [p.text for p in self.corpus.passages]
        return self._texts

    @classmethod
    def build(cls, corpus: Corpus, *, with_dense: bool = True) -> Index:
        texts = [p.text for p in corpus.passages]
        bm25 = BM25().index(texts)
        dense = Dense().index(texts) if with_dense else None
        idx = cls(corpus=corpus, bm25=bm25, dense=dense)
        idx._texts = texts
        return idx


def bm25_ranker(index: Index) -> Ranker:
    return lambda q, k: [d for d, _ in index.bm25.search(q, k)]


def dense_ranker(index: Index) -> Ranker:
    def rank(q: str, k: int) -> list[int]:
        assert index.dense is not None
        return [d for d, _ in index.dense.search(q, k)]

    return rank


def hybrid_ranker(index: Index, *, pool: int = 50) -> Ranker:
    """BM25 + dense, fused by RRF.

    Both halves are over-retrieved to `pool` before fusing: RRF works on rank
    position, so a document that only one retriever finds still needs to be
    *in* that retriever's list to contribute.
    """

    def rank(q: str, k: int) -> list[int]:
        assert index.dense is not None
        sparse = index.bm25.search(q, pool)
        dense = index.dense.search(q, pool)
        return [d for d, _ in rrf([sparse, dense], k=k)]

    return rank


def rerank_ranker(index: Index, base: Ranker, reranker: Reranker, *, pool: int = 50) -> Ranker:
    """Over-retrieve with `base`, then reorder with a cross-encoder."""

    def rank(q: str, k: int) -> list[int]:
        candidates = base(q, pool)
        pairs = [(d, index.texts[d]) for d in candidates]
        return [d for d, _ in reranker.rerank(q, pairs, k=k)]

    return rank


def run(
    ranker: Ranker,
    queries: Sequence[Query],
    *,
    depth: int = 20,
    ks: Sequence[int] = (1, 2, 5, 10, 20),
) -> Scores:
    """Score one strategy over the query set."""
    t0 = time.time()
    rankings = [(ranker(q.question, depth), q.gold_doc_ids) for q in queries]
    return evaluate(rankings, ks=ks, seconds=time.time() - t0)
