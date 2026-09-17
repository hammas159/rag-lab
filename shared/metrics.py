"""Retrieval metrics, computed against known-gold document ids.

None of this needs a language model. A RAG variant is a retrieval technique, and
whether it put the right passage in front of the model is answerable on its own.
Measuring end-to-end answer quality instead adds the generator's noise on top of
the effect being measured, which is why so many RAG comparisons disagree.

`recall_at_k` here is **all-gold recall**: a HotpotQA question needs both of its
supporting paragraphs, so retrieving one of two scores 0.5, not 1.0. The stricter
`answerable_at_k` asks the question a pipeline actually cares about -- were *all*
the required passages retrieved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def recall_at_k(ranked: Sequence[int], gold: frozenset[int], k: int) -> float:
    """Fraction of gold documents appearing in the top k."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def answerable_at_k(ranked: Sequence[int], gold: frozenset[int], k: int) -> bool:
    """True only if *every* gold document is in the top k.

    For multi-hop questions this is the honest measure: a pipeline that retrieves
    one of the two bridging paragraphs cannot answer, however good its recall
    number looks.
    """
    return bool(gold) and gold <= set(ranked[:k])


def mrr(ranked: Sequence[int], gold: frozenset[int]) -> float:
    """Reciprocal rank of the first gold document."""
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], gold: frozenset[int], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class Scores:
    n: int = 0
    recall: dict[int, float] = None  # type: ignore[assignment]
    answerable: dict[int, float] = None  # type: ignore[assignment]
    ndcg: dict[int, float] = None  # type: ignore[assignment]
    mrr: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mrr": round(self.mrr, 4),
            "recall": {k: round(v, 4) for k, v in sorted(self.recall.items())},
            "answerable": {k: round(v, 4) for k, v in sorted(self.answerable.items())},
            "ndcg": {k: round(v, 4) for k, v in sorted(self.ndcg.items())},
            "seconds": round(self.seconds, 2),
        }


def evaluate(
    rankings: Sequence[tuple[Sequence[int], frozenset[int]]],
    ks: Sequence[int] = (1, 2, 5, 10, 20),
    seconds: float = 0.0,
) -> Scores:
    """Aggregate one run. `rankings` is (ranked_doc_ids, gold_doc_ids) per query."""
    n = len(rankings)
    if n == 0:
        return Scores(0, {k: 0.0 for k in ks}, {k: 0.0 for k in ks}, {k: 0.0 for k in ks})
    return Scores(
        n=n,
        recall={k: sum(recall_at_k(r, g, k) for r, g in rankings) / n for k in ks},
        answerable={k: sum(answerable_at_k(r, g, k) for r, g in rankings) / n for k in ks},
        ndcg={k: sum(ndcg_at_k(r, g, k) for r, g in rankings) / n for k in ks},
        mrr=sum(mrr(r, g) for r, g in rankings) / n,
        seconds=seconds,
    )
