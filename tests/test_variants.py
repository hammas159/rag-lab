"""Tests for the variant implementations in projects/.

Project 06 shipped a wrong answer twice before a right one -- both times because
the *merge* was wrong, not the technique, and neither failure crashed. Appending
candidates after a full-depth list silently makes them unreachable; fusing a weak
ranking equal-weight silently destroys a strong one. Both look like findings.
These tests pin the behaviours that made those bugs invisible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.retrieval import rrf  # noqa: E402


def _load(project: str):
    """Import a project's run.py without executing main()."""
    path = ROOT / "projects" / project / "run.py"
    spec = importlib.util.spec_from_file_location(f"proj_{project}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestMergeSemantics:
    """The class of bug that produced two false findings in project 06."""

    def test_appending_after_a_full_list_cannot_change_the_top_k(self):
        """This is why 'identical to baseline' was a bug, not a result.

        With depth 20 and 50 base hits, anything appended afterwards is
        unreachable -- so the experiment was guaranteed to score the baseline
        before it ran.
        """
        base = list(range(50))
        extra = [900, 901, 902]
        merged = base + [d for d in extra if d not in base]
        assert merged[:20] == base[:20]
        assert not set(extra) & set(merged[:20])

    def test_equal_weight_fusion_lets_a_weak_ranking_displace_a_strong_one(self):
        """And this is why the equal-weight version scored 0.406.

        RRF ranks by position, not quality. A ranking of junk contributes the
        same positional weight as a ranking of gold.
        """
        strong = [(i, 1.0) for i in range(10)]  # 0..9 are the good documents
        junk = [(500 + i, 1.0) for i in range(10)]
        fused = [d for d, _ in rrf([strong, junk], k=10)]
        displaced = [d for d in fused if d >= 500]
        assert displaced, "junk should intrude, which is exactly the hazard"
        assert len(set(fused) & set(range(10))) < 10


class TestRetrievalUnit:
    """Project 01: units must map back to documents correctly."""

    def setup_method(self):
        self.m = _load("01_retrieval_unit")

    def test_units_collapse_to_owning_documents(self):
        owner = [7, 7, 7, 3, 3]
        assert self.m.to_docs([0, 1, 2, 3], owner, k=5) == [7, 3]

    def test_collapse_preserves_first_hit_order(self):
        """A document's rank is its *best* unit's rank, not its last."""
        owner = [1, 2, 1]
        assert self.m.to_docs([2, 1, 0], owner, k=5) == [1, 2]

    def test_collapse_respects_k(self):
        owner = [1, 2, 3, 4]
        assert self.m.to_docs([0, 1, 2, 3], owner, k=2) == [1, 2]


class TestRagFusionVariants:
    """Project 02: rewrites must be deterministic and never empty."""

    def setup_method(self):
        self.m = _load("02_rag_fusion")

    def test_keywords_drops_stopwords(self):
        out = self.m.variant_keywords("Who is the director of the film?")
        assert "the" not in out.split()
        assert "director" in out

    def test_entities_finds_capitalised_names(self):
        out = self.m.variant_entities("Were Scott Derrickson and Ed Wood Americans?")
        assert "Scott Derrickson" in out

    def test_entities_ignores_the_leading_capital(self):
        """'Were' starts the sentence; it is not an entity."""
        assert "Were" not in self.m.variant_entities("Were Scott Derrickson American?")

    def test_every_variant_falls_back_rather_than_returning_empty(self):
        """An empty query retrieves nothing and silently drops a fusion arm."""
        for q in ("what happened", "?", "a of the"):
            assert self.m.variant_entities(q).strip()
            assert self.m.variant_quoted(q).strip()

    def test_split_returns_nothing_when_there_is_one_clause(self):
        assert self.m.variant_split("Who directed Ed Wood?") == []


class TestMultiHop:
    """Project 03: hop-2 must add vocabulary the question lacked."""

    def setup_method(self):
        self.m = _load("03_iterative_multihop")

    def test_hop2_adds_terms_absent_from_the_question(self):
        q = "Who directed the film?"
        out = self.m.hop2_query(q, ["Tim Burton directed Ed Wood in 1994"])
        assert "burton" in out.lower()

    def test_hop2_does_not_repeat_question_terms(self):
        """Repeating them just re-finds the same documents."""
        q = "Who directed the film?"
        out = self.m.hop2_query(q, ["directed directed directed"])
        assert out.lower().split().count("directed") == 1

    def test_hop2_keeps_the_original_question(self):
        q = "Who directed the film?"
        assert self.m.hop2_query(q, ["x y z"]).startswith(q)

    @pytest.mark.parametrize(
        "question",
        [
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "Which came first, the book or the film?",
            "Are both directors American?",
        ],
    )
    def test_comparison_cues_are_detected(self, question):
        assert self.m.looks_like_comparison(question)

    def test_a_plain_bridge_question_is_not_flagged(self):
        assert not self.m.looks_like_comparison(
            "Who directed the 1994 biographical film about a filmmaker?"
        )


class TestEntityGraph:
    """Project 07: extraction and hub-dropping."""

    def setup_method(self):
        self.m = _load("07_entity_graph")

    def test_extracts_multiword_names(self):
        assert "scott derrickson" in self.m.entities("Scott Derrickson is a director.")

    def test_ignores_short_and_shouting_spans(self):
        found = self.m.entities("The USA and BBC are Big.")
        assert "usa" not in found and "bbc" not in found
