"""Retrievers: BM25, dense, RRF fusion, cross-encoder reranking.

BM25 is implemented here rather than imported. It is forty lines, it removes a
dependency from the sparse half of every experiment in this repo, and -- since
nlp-lab measured BM25 landing within 8 points of a pretrained neural embedding
for a fraction of the indexing cost -- it is the baseline every variant in this
lab has to beat. A baseline worth taking seriously is worth owning.

Dense retrieval uses `BAAI/bge-small-en-v1.5`, which is small enough to index
tens of thousands of passages on a laptop GPU and is not trained on HotpotQA.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")

# Kept deliberately short. An aggressive stoplist helps BM25 on keyword queries
# and hurts it on natural-language questions, and these are questions.
_STOP = frozenset(
    # fmt: off
    [
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "for",
        "with",
        "as",
        "by",
        "that",
        "this",
        "it",
        "its",
        "from",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "how",
    ]
    # fmt: on
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = _WORD.findall(text.lower())
    return [t for t in toks if t not in _STOP] if drop_stopwords else toks


@dataclass
class BM25:
    """Okapi BM25 over a fixed corpus.

    `k1` controls term-frequency saturation, `b` how hard long documents are
    penalised. The defaults are the usual ones; they are exposed because the
    retrieval-unit experiment changes document length by an order of magnitude,
    and a length-normalisation constant tuned for paragraphs is not obviously
    right for whole documents.
    """

    k1: float = 1.5
    b: float = 0.75
    doc_len: list[int] = field(default_factory=list)
    avg_len: float = 0.0
    # term -> [(doc_id, term_frequency), ...]. Scoring only ever touches the
    # documents that actually contain a query term; scanning the whole corpus per
    # term instead turns a 1500-query run into hundreds of millions of operations.
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    _idf_cache: dict[str, float] = field(default_factory=dict)

    def index(self, texts: Sequence[str]) -> BM25:
        self.postings = {}
        self.doc_len = []
        self._idf_cache = {}
        for doc_id, text in enumerate(texts):
            toks = tokenize(text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((doc_id, tf))
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        n = len(self.doc_len)
        for term, plist in self.postings.items():
            df = len(plist)
            # Robertson/Sparck-Jones idf with the +1 that keeps it non-negative.
            self._idf_cache[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        avg = self.avg_len or 1.0
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self._idf_cache[term]
            for doc_id, f in plist:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / avg)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


@dataclass
class Dense:
    """Dense retrieval over normalised embeddings, cosine via dot product."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 128
    _model: object | None = None
    _emb: object | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, texts: Sequence[str]) -> Dense:
        model = self._load()
        self._emb = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[tuple[int, float]]]:
        import numpy as np

        model = self._load()
        # bge asks for this prefix on the query side only; skipping it costs a
        # few points of recall and is an easy mistake to make.
        q = model.encode(
            [f"Represent this sentence for searching relevant passages: {x}" for x in queries],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = q @ self._emb.T  # type: ignore[union-attr]
        out = []
        for row in sims:
            top = np.argpartition(-row, min(k, len(row) - 1))[:k]
            top = top[np.argsort(-row[top])]
            out.append([(int(i), float(row[i])) for i in top])
        return out


def rrf(
    rankings: Sequence[Sequence[tuple[int, float]]], k: int = 10, c: int = 60
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Fuses rankings by position rather than score, so a BM25 score of 14.2 and a
    cosine of 0.83 never have to be made commensurable -- which is the whole
    reason RRF is used instead of weighted score blending.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])[:k]


@dataclass
class Reranker:
    """Cross-encoder reranking of an over-retrieved candidate list."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 64
    _model: object | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[tuple[int, str]], k: int = 10
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        model = self._load()
        scores = model.predict(
            [(query, text) for _, text in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        pairs = [(doc_id, float(s)) for (doc_id, _), s in zip(candidates, scores, strict=False)]
        return sorted(pairs, key=lambda kv: -kv[1])[:k]
