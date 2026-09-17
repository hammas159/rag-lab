"""05 - Corrective RAG: knowing when retrieval failed.

CRAG's structure is a retrieval evaluator that grades what came back and triggers
a correction when the grade is poor. The evaluator is usually a model. But the
retriever already emits a signal about its own confidence -- the score it gave
the top document, and how far that score sits above the rest -- and nothing has
to read the text to use it.

The question this asks is the one that decides whether CRAG is worth building:
**can retrieval tell, from its own scores, that it has failed?** If it cannot,
the correction fires at random and a smarter correction will not save it.

Three signals, none of them a model:

    top_score     the fused score of rank 1
    margin        rank 1 minus rank 2 -- a flat ranking means nothing stood out
    agreement     did BM25 and dense independently put the same doc on top

Corrections are cheap and rule-based: widen the pool, or fall back to the
iterative second hop from project 03.

    python run.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.corpus import load  # noqa: E402
from shared.metrics import answerable_at_k, evaluate  # noqa: E402
from shared.pipeline import Index  # noqa: E402
from shared.retrieval import rrf, tokenize  # noqa: E402

HERE = Path(__file__).resolve().parent
N_QUESTIONS = 1500
DEPTH = 20
POOL = 50


def fused(index: Index, query: str, k: int, pool: int = POOL):
    sparse = index.bm25.search(query, pool)
    dense = index.dense.search(query, pool)  # type: ignore[union-attr]
    return rrf([sparse, dense], k=k), sparse, dense


def signals(ranked, sparse, dense) -> dict:
    top = ranked[0][1] if ranked else 0.0
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "top_score": top,
        "margin": top - second,
        "agreement": bool(sparse and dense and sparse[0][0] == dense[0][0]),
    }


def hop2(index: Index, corpus, question: str, seeds: list[int], max_terms: int = 40) -> str:
    asked = set(tokenize(question))
    extra: list[str] = []
    for doc_id in seeds:
        for term in tokenize(corpus.get(doc_id).text):
            if term not in asked and term not in extra:
                extra.append(term)
            if len(extra) >= max_terms:
                break
        if len(extra) >= max_terms:
            break
    return f"{question} {' '.join(extra)}"


def main() -> None:
    corpus, queries = load(N_QUESTIONS)
    index = Index.build(corpus)
    print(f"corpus: {len(corpus)} passages, queries: {len(queries)}\n")

    # --- can the scores tell a good run from a bad one? --------------------------
    rows = []
    for q in queries:
        ranked, sparse, dense = fused(index, q.question, DEPTH)
        docs = [d for d, _ in ranked]
        rows.append(
            {
                "q": q,
                "docs": docs,
                "ok": answerable_at_k(docs, q.gold_doc_ids, 10),
                **signals(ranked, sparse, dense),
            }
        )

    good = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    diagnostic = {
        "answerable@10": round(len(good) / len(rows), 4),
        "top_score": {
            "when_answerable": round(statistics.fmean(r["top_score"] for r in good), 5),
            "when_not": round(statistics.fmean(r["top_score"] for r in bad), 5),
        },
        "margin": {
            "when_answerable": round(statistics.fmean(r["margin"] for r in good), 5),
            "when_not": round(statistics.fmean(r["margin"] for r in bad), 5),
        },
        "agreement_rate": {
            "when_answerable": round(sum(r["agreement"] for r in good) / len(good), 4),
            "when_not": round(sum(r["agreement"] for r in bad) / len(bad), 4),
        },
    }
    print("can retrieval detect its own failure?")
    for key in ("top_score", "margin", "agreement_rate"):
        v = diagnostic[key]
        print(
            f"  {key:16} answerable={v['when_answerable']:.4f}   not answerable={v['when_not']:.4f}"
        )
    print()

    # A trigger worth using must fire more often on the failures than the successes.
    thresholds = {
        "always correct": lambda r: True,
        "never correct (baseline)": lambda r: False,
        "low agreement": lambda r: not r["agreement"],
        "low margin": lambda r: r["margin"] < statistics.median(x["margin"] for x in rows),
    }

    results = {"diagnostic": diagnostic, "strategies": {}}
    for name, trigger in thresholds.items():
        t0 = time.time()
        ranked_out = []
        fired = 0
        for r in rows:
            q = r["q"]
            if trigger(r):
                fired += 1
                seeds = r["docs"][:2]
                corrected, _, _ = fused(index, hop2(index, corpus, q.question, seeds), DEPTH)
                merged = list(seeds)
                for d, _ in corrected:
                    if d not in merged:
                        merged.append(d)
                for d in r["docs"]:
                    if d not in merged:
                        merged.append(d)
                ranked_out.append((merged[:DEPTH], q.gold_doc_ids))
            else:
                ranked_out.append((r["docs"], q.gold_doc_ids))
        d = evaluate(ranked_out, seconds=time.time() - t0).as_dict()
        d["corrections_fired"] = fired
        d["fire_rate"] = round(fired / len(rows), 4)
        # Precision of the trigger: of the queries it fired on, how many needed it?
        needed = sum(1 for r in rows if trigger(r) and not r["ok"])
        d["trigger_precision"] = round(needed / fired, 4) if fired else None
        results["strategies"][name] = d
        print(
            f"{name:26} answerable@10={d['answerable'][10]:.3f}  "
            f"fired={d['fire_rate']:.0%}  trigger_precision={d['trigger_precision']}"
        )

    out = HERE / "results.json"
    out.write_text(
        json.dumps({"n_questions": len(queries), "depth": DEPTH, **results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()
