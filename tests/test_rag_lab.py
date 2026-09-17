"""Tests for rag-lab.

Every conclusion in this repo is a number produced by `metrics.py` over rankings
produced by `retrieval.py`. Both are worth testing on cases where the right
answer is known by hand, because a metric bug does not crash -- it quietly
changes the finding.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.corpus import Corpus  # noqa: E402
from shared.metrics import (  # noqa: E402
    answerable_at_k,
    evaluate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from shared.retrieval import BM25, rrf, tokenize  # noqa: E402


class TestRecallVsAnswerable:
    """The distinction the whole lab rests on."""

    def test_half_the_gold_is_half_recall_but_not_answerable(self):
        """A multi-hop question with one of two passages cannot be answered.

        recall@k rewards partial retrieval; answerable@k does not. Conflating
        them is what makes a pipeline look 17 points better than it is.
        """
        ranked, gold = [1, 9, 8], frozenset({1, 2})
        assert recall_at_k(ranked, gold, 3) == 0.5
        assert answerable_at_k(ranked, gold, 3) is False

    def test_all_gold_present_is_answerable(self):
        assert answerable_at_k([5, 1, 2], frozenset({1, 2}), 3) is True

    def test_answerable_respects_the_cutoff(self):
        """Gold at rank 4 does not count for k=3."""
        assert answerable_at_k([1, 7, 8, 2], frozenset({1, 2}), 3) is False
        assert answerable_at_k([1, 7, 8, 2], frozenset({1, 2}), 4) is True

    def test_empty_gold_is_never_answerable(self):
        assert answerable_at_k([1, 2], frozenset(), 2) is False
        assert recall_at_k([1, 2], frozenset(), 2) == 0.0


class TestRankingMetrics:
    def test_mrr_uses_the_first_gold_hit(self):
        assert mrr([9, 8, 1, 2], frozenset({1, 2})) == pytest.approx(1 / 3)

    def test_mrr_is_zero_when_nothing_is_found(self):
        assert mrr([7, 8, 9], frozenset({1})) == 0.0

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        assert ndcg_at_k([1, 2, 9], frozenset({1, 2}), 3) == pytest.approx(1.0)

    def test_ndcg_penalises_position(self):
        """Same documents, worse order, lower score."""
        good = ndcg_at_k([1, 2, 9], frozenset({1, 2}), 3)
        bad = ndcg_at_k([9, 1, 2], frozenset({1, 2}), 3)
        assert bad < good

    def test_ndcg_matches_the_formula_by_hand(self):
        # gold at ranks 2 and 3: dcg = 1/log2(3) + 1/log2(4)
        # ideal (ranks 1,2)    : 1/log2(2) + 1/log2(3)
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        ideal = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k([9, 1, 2], frozenset({1, 2}), 3) == pytest.approx(dcg / ideal)


class TestEvaluate:
    def test_aggregates_across_queries(self):
        scores = evaluate([([1, 2], frozenset({1, 2})), ([9, 8], frozenset({1, 2}))], ks=(2,))
        assert scores.n == 2
        assert scores.recall[2] == pytest.approx(0.5)  # 1.0 and 0.0
        assert scores.answerable[2] == pytest.approx(0.5)

    def test_empty_run_does_not_divide_by_zero(self):
        assert evaluate([], ks=(1, 5)).n == 0


class TestTokenizer:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Ed Wood's film, 1994!", drop_stopwords=False) == [
            "ed",
            "wood",
            "s",
            "film",
            "1994",
        ]

    def test_stopwords_are_dropped_by_default(self):
        assert "the" not in tokenize("the director of the film")

    def test_stopwords_can_be_kept(self):
        assert "the" in tokenize("the director", drop_stopwords=False)


class TestBM25:
    def test_ranks_the_document_containing_the_query_term_first(self):
        bm = BM25().index(["cats sleep often", "dogs bark loudly", "cats chase dogs"])
        top = bm.search("cats", k=3)
        assert top[0][0] in (0, 2)

    def test_a_rare_term_outranks_a_common_one(self):
        """idf is the whole point: a term in every document carries no signal."""
        docs = ["common word here", "common word there", "common word zebra"]
        bm = BM25().index(docs)
        assert bm.search("zebra", k=1)[0][0] == 2

    def test_unknown_terms_return_nothing_rather_than_crashing(self):
        bm = BM25().index(["alpha beta"])
        assert bm.search("nonexistentterm", k=5) == []

    def test_length_normalisation_prefers_the_shorter_document(self):
        """Same term count, more padding: b>0 should push the long one down."""
        short = "quantum"
        long = "quantum " + " ".join(f"filler{i}" for i in range(200))
        bm = BM25(b=0.75).index([long, short])
        assert bm.search("quantum", k=2)[0][0] == 1

    def test_inverted_index_covers_every_term(self):
        bm = BM25().index(["alpha beta", "beta gamma"])
        assert set(bm.postings) == {"alpha", "beta", "gamma"}
        assert sorted(d for d, _ in bm.postings["beta"]) == [0, 1]


class TestRRF:
    def test_a_document_ranked_well_by_both_wins(self):
        a = [(1, 9.0), (2, 8.0), (3, 7.0)]
        b = [(1, 0.9), (3, 0.8), (2, 0.7)]
        assert rrf([a, b], k=3)[0][0] == 1

    def test_fusion_ignores_score_scale(self):
        """RRF exists so a BM25 score of 14.2 and a cosine of 0.83 never have to
        be made commensurable. Rescaling one side must change nothing."""
        a = [(1, 9.0), (2, 8.0)]
        b = [(2, 0.9), (1, 0.8)]
        scaled = [(d, s * 1000) for d, s in a]
        assert [d for d, _ in rrf([a, b], k=2)] == [d for d, _ in rrf([scaled, b], k=2)]

    def test_a_document_only_one_retriever_found_still_appears(self):
        fused = [d for d, _ in rrf([[(1, 1.0)], [(2, 1.0)]], k=2)]
        assert set(fused) == {1, 2}


class TestCorpus:
    def test_the_same_title_is_one_document(self):
        """De-duplication is by title. Treating repeats as separate documents
        would split a query's gold label across several ids."""
        c = Corpus()
        first = c.add("Ed Wood", ["a"])
        again = c.add("Ed Wood", ["a"])
        assert first == again and len(c) == 1

    def test_different_titles_are_different_documents(self):
        c = Corpus()
        assert c.add("A", ["x"]) != c.add("B", ["y"])
        assert len(c) == 2

    def test_summary_reports_real_token_counts(self):
        c = Corpus()
        c.add("T", ["one two three"])
        assert c.summary()["passages"] == 1
        assert c.summary()["total_tokens"] == 3
