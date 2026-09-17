"""03 - Multi-hop retrieval without a reasoner.

A bridge question hides its second document behind the first: "the director of
the film that X starred in" cannot match the director's page, because the
question never names him. The usual fix is an agent that reads hop one and writes
a new query for hop two -- an LLM call per hop.

This asks whether the *retrieval* half of that loop carries the weight on its
own. Round two's query is round one's question plus the text already retrieved.
No model reasons about what is missing; the retrieved passage simply supplies the
vocabulary the question lacked.

    round 1   retrieve on the question
    round 2   retrieve on question + top-n round-1 text, excluding what was kept
    merge     round-1 results first, then round-2's best

HotpotQA labels each question `bridge` or `comparison`. A comparison question
names both entities up front and should gain nothing from a second hop, so the
label lets the effect be reported per type instead of averaged into mush -- and a
rule-based classifier is measured against that label too, since routing is only
useful if the route can be predicted.

    python run.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.corpus import load  # noqa: E402
from shared.metrics import evaluate  # noqa: E402
from shared.pipeline import Index  # noqa: E402
from shared.retrieval import rrf, tokenize  # noqa: E402

HERE = Path(__file__).resolve().parent
N_QUESTIONS = 1500
DEPTH = 20
POOL = 50

_COMPARISON_CUES = (
    "both",
    "same",
    "which came first",
    "who is older",
    "who was born first",
    "are both",
    "were both",
    "or ",
    " than ",
    "more",
    "less",
    "earlier",
    "later",
)


def looks_like_comparison(question: str) -> bool:
    """Rule-based bridge/comparison classifier. No model, no training."""
    q = question.lower()
    if any(cue in q for cue in _COMPARISON_CUES):
        return True
    # "A and B" where both look like names is almost always a comparison.
    caps = re.findall(r"\b[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*", question[1:])
    return len(caps) >= 2 and (" and " in q)


def hop2_query(question: str, seed_texts: list[str], max_terms: int = 40) -> str:
    """Round-2 query: the question, plus vocabulary from round-1 passages.

    Terms already in the question are dropped -- repeating them re-finds the same
    documents. What is left is the bridge: names and nouns the question could not
    have contained.
    """
    asked = set(tokenize(question))
    extra: list[str] = []
    for text in seed_texts:
        for term in tokenize(text):
            if term not in asked and term not in extra:
                extra.append(term)
            if len(extra) >= max_terms:
                break
        if len(extra) >= max_terms:
            break
    return f"{question} {' '.join(extra)}"


def retrieve(index: Index, query: str, k: int) -> list[int]:
    fused = rrf([index.bm25.search(query, POOL), index.dense.search(query, POOL)], k=k)  # type: ignore[union-attr]
    return [d for d, _ in fused]


def main() -> None:
    corpus, queries = load(N_QUESTIONS)
    index = Index.build(corpus)
    print(f"corpus: {len(corpus)} passages, queries: {len(queries)}\n")

    # --- the classifier, measured before it is used ------------------------------
    tp = sum(looks_like_comparison(q.question) and q.hop_type == "comparison" for q in queries)
    fp = sum(looks_like_comparison(q.question) and q.hop_type == "bridge" for q in queries)
    fn = sum(not looks_like_comparison(q.question) and q.hop_type == "comparison" for q in queries)
    tn = sum(not looks_like_comparison(q.question) and q.hop_type == "bridge" for q in queries)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    classifier = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "accuracy": round((tp + tn) / len(queries), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    print(f"rule-based comparison classifier: {classifier}\n")

    strategies = {
        "single hop": lambda q: retrieve(index, q.question, DEPTH),
        "two hops (always)": lambda q: two_hop(index, q, corpus),
        "two hops (bridge only)": lambda q: (
            two_hop(index, q, corpus)
            if not looks_like_comparison(q.question)
            else retrieve(index, q.question, DEPTH)
        ),
    }

    results = {}
    for name, fn_strategy in strategies.items():
        t0 = time.time()
        ranked = [(fn_strategy(q), q.gold_doc_ids) for q in queries]
        elapsed = time.time() - t0
        overall = evaluate(ranked, seconds=elapsed).as_dict()
        by_type = {}
        for hop_type in ("bridge", "comparison"):
            subset = [
                (r, g) for (r, g), q in zip(ranked, queries, strict=False) if q.hop_type == hop_type
            ]
            by_type[hop_type] = evaluate(subset).as_dict()
        results[name] = {"overall": overall, "by_type": by_type}
        print(
            f"{name:24} answerable@10: overall={overall['answerable'][10]:.3f}  "
            f"bridge={by_type['bridge']['answerable'][10]:.3f}  "
            f"comparison={by_type['comparison']['answerable'][10]:.3f}  ({overall['seconds']}s)"
        )

    out = HERE / "results.json"
    out.write_text(
        json.dumps(
            {
                "n_questions": len(queries),
                "depth": DEPTH,
                "classifier": classifier,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out.name}")


def two_hop(index: Index, q, corpus, seed_n: int = 2) -> list[int]:
    first = retrieve(index, q.question, DEPTH)
    seeds = [corpus.get(d).text for d in first[:seed_n]]
    second = retrieve(index, hop2_query(q.question, seeds), DEPTH)
    merged = list(first[:seed_n])
    for d in second:
        if d not in merged:
            merged.append(d)
    for d in first:
        if d not in merged:
            merged.append(d)
    return merged[:DEPTH]


if __name__ == "__main__":
    main()
