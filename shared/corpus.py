"""A real retrieval corpus with ground truth, built from HotpotQA.

HotpotQA's distractor split gives every question ten paragraphs: two that the
annotators actually used to answer it, and eight plausible distractors pulled
from the same Wikipedia neighbourhood. That is the part worth having. Most RAG
demos invent a corpus where the right answer is the only thing that mentions the
topic; here the wrong paragraphs are about the same people and places, which is
what makes retrieval hard.

Pooling the paragraphs of N questions and de-duplicating by title gives a corpus
of tens of thousands of real passages, and for each question the titles that are
known to be sufficient. That is a retrieval benchmark: no model has to generate
anything for the score to be meaningful.

    corpus, queries = load(n_questions=1500)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / ".corpus_cache"
DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"


@dataclass(frozen=True)
class Passage:
    """One Wikipedia paragraph. `title` is the identity used by the gold labels."""

    doc_id: int
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences).strip()

    @property
    def n_tokens(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Query:
    qid: str
    question: str
    answer: str
    gold_doc_ids: frozenset[int]
    level: str  # easy | medium | hard
    hop_type: str  # bridge | comparison

    @property
    def n_gold(self) -> int:
        return len(self.gold_doc_ids)


@dataclass
class Corpus:
    passages: list[Passage] = field(default_factory=list)
    _by_title: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.passages)

    def add(self, title: str, sentences: list[str]) -> int:
        """Return the doc_id for this title, creating it on first sight.

        De-duplication is by title, not by text. The same Wikipedia article can
        appear for many questions with an identical paragraph; treating those as
        separate documents would inflate the corpus and, worse, split a query's
        gold label across several ids.
        """
        if title in self._by_title:
            return self._by_title[title]
        doc_id = len(self.passages)
        self.passages.append(Passage(doc_id, title, tuple(sentences)))
        self._by_title[title] = doc_id
        return doc_id

    def get(self, doc_id: int) -> Passage:
        return self.passages[doc_id]

    def summary(self) -> dict:
        toks = [p.n_tokens for p in self.passages]
        toks.sort()
        return {
            "passages": len(self.passages),
            "total_tokens": sum(toks),
            "median_tokens": toks[len(toks) // 2] if toks else 0,
            "p95_tokens": toks[int(len(toks) * 0.95)] if toks else 0,
            "max_tokens": toks[-1] if toks else 0,
        }


def _fingerprint(n_questions: int, seed: int) -> str:
    return hashlib.sha256(f"{DATASET}|{CONFIG}|{SPLIT}|{n_questions}|{seed}".encode()).hexdigest()[
        :16
    ]


@lru_cache(maxsize=4)
def load(n_questions: int = 1500, seed: int = 17) -> tuple[Corpus, tuple[Query, ...]]:
    """Build (or load) the corpus and its queries.

    Sampling is seeded and the sample is cached to disk, so every project in this
    repo measures against byte-identical data. A variant that looked better on a
    different sample would not be a finding.
    """
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"hotpot_{_fingerprint(n_questions, seed)}.json"
    if path.exists():
        return _from_json(json.loads(path.read_text(encoding="utf-8")))

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from datasets import load_dataset  # imported late: the cache path needs no datasets

    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    if n_questions < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_questions))

    corpus = Corpus()
    queries: list[Query] = []
    for row in ds:
        ctx = row["context"]
        title_to_id = {}
        for title, sents in zip(ctx["title"], ctx["sentences"], strict=False):
            title_to_id[title] = corpus.add(title, list(sents))

        gold = {title_to_id[t] for t in row["supporting_facts"]["title"] if t in title_to_id}
        if not gold:
            # A question whose gold paragraph is not among its own context is
            # unanswerable from this corpus; scoring it would penalise every
            # retriever equally for something none of them can fix.
            continue
        queries.append(
            Query(
                qid=row["id"],
                question=row["question"],
                answer=row["answer"],
                gold_doc_ids=frozenset(gold),
                level=row["level"],
                hop_type=row["type"],
            )
        )

    payload = _to_json(corpus, queries)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return corpus, tuple(queries)


def _to_json(corpus: Corpus, queries: list[Query]) -> dict:
    return {
        "passages": [[p.title, list(p.sentences)] for p in corpus.passages],
        "queries": [
            [q.qid, q.question, q.answer, sorted(q.gold_doc_ids), q.level, q.hop_type]
            for q in queries
        ],
    }


def _from_json(payload: dict) -> tuple[Corpus, tuple[Query, ...]]:
    corpus = Corpus()
    for title, sentences in payload["passages"]:
        corpus.add(title, sentences)
    queries = tuple(
        Query(qid, question, answer, frozenset(gold), level, hop_type)
        for qid, question, answer, gold, level, hop_type in payload["queries"]
    )
    return corpus, queries
