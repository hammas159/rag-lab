"""04 - HyDE, and the 1970s technique that does the same job for free.

HyDE's insight is that a question and its answer do not look alike. "Who directed
the film Ed Wood?" shares almost no vocabulary with a paragraph about Tim Burton.
So HyDE asks a model to *invent* an answer, embeds that hallucination, and
searches with it -- a fake answer is shaped like a real one, even when its facts
are wrong.

Pseudo-relevance feedback had the same idea in 1971 and needs no model: retrieve
once, assume the top results are relevant, and enrich the query with their terms.
Both replace the question with something answer-shaped. One costs a generation
per query; the other costs a second index lookup.

This project runs both against the same queries and the same corpus, so the
question is not "does HyDE work" but "does HyDE beat the free version".

    hyde   qwen2.5:7b-instruct writes a short passage answering the question
    prf    top-3 retrieved passages donate their highest-idf terms
    both   the union, fused with RRF

    python run.py              # full run
    python run.py --limit 300  # shorter, for a first look
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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
MODEL = "qwen2.5:7b-instruct"
OLLAMA = "http://127.0.0.1:11434/api/generate"

HYDE_PROMPT = (
    "Write one short factual paragraph that would answer this question. "
    "Do not say you are unsure, do not ask for clarification, do not add "
    "commentary. Two or three sentences of encyclopedia-style prose only.\n\n"
    "Question: {q}\n\nPassage:"
)


def retrieve(index: Index, query: str, k: int, pool: int = POOL) -> list[tuple[int, float]]:
    fused = rrf([index.bm25.search(query, pool), index.dense.search(query, pool)], k=k)  # type: ignore[union-attr]
    return fused


def prf_query(
    index: Index, corpus, question: str, *, feedback_docs: int = 3, terms: int = 25
) -> str:
    """Rocchio-style pseudo-relevance feedback, no model involved.

    Take the top documents on faith, and add the terms that are frequent in them
    and absent from the question. Those terms are what the question could not
    say: names, dates, the vocabulary of the answer rather than the ask.
    """
    top = retrieve(index, question, feedback_docs)
    asked = set(tokenize(question))
    counts: Counter = Counter()
    for doc_id, _ in top:
        for term in tokenize(corpus.get(doc_id).text):
            if term not in asked:
                counts[term] += 1
    extra = [t for t, _ in counts.most_common(terms)]
    return f"{question} {' '.join(extra)}"


def hyde_passage(question: str, timeout: float = 120.0) -> str:
    import httpx

    r = httpx.post(
        OLLAMA,
        json={
            "model": MODEL,
            "prompt": HYDE_PROMPT.format(q=question),
            "stream": False,
            "options": {"temperature": 0.0, "seed": 0, "num_predict": 160},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N_QUESTIONS)
    args = ap.parse_args()

    corpus, queries = load(N_QUESTIONS)
    queries = queries[: args.limit]
    index = Index.build(corpus)
    print(f"corpus: {len(corpus)} passages, queries: {len(queries)}\n")

    # --- generate the HyDE passages once, cached to disk --------------------------
    cache_path = HERE / f"hyde_passages_{args.limit}.json"
    if cache_path.exists():
        hyde_docs = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"loaded {len(hyde_docs)} cached HyDE passages")
    else:
        hyde_docs = {}
        t0 = time.time()
        for i, q in enumerate(queries, 1):
            try:
                hyde_docs[q.qid] = hyde_passage(q.question)
            except Exception as exc:  # noqa: BLE001 - one dead call must not lose the run
                hyde_docs[q.qid] = ""
                print(f"  [{i}] generation failed: {type(exc).__name__}")
            if i % 100 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(queries)} generated ({rate:.2f}s each)")
        cache_path.write_text(json.dumps(hyde_docs), encoding="utf-8")
        print(f"generated {len(hyde_docs)} passages in {time.time() - t0:.0f}s")

    empty = sum(1 for v in hyde_docs.values() if not v.strip())
    print(f"empty generations: {empty}\n")

    strategies = {
        "baseline (question only)": lambda q: [d for d, _ in retrieve(index, q.question, DEPTH)],
        "PRF (no model)": lambda q: [
            d for d, _ in retrieve(index, prf_query(index, corpus, q.question), DEPTH)
        ],
        "HyDE (7b-instruct)": lambda q: [
            d for d, _ in retrieve(index, f"{q.question} {hyde_docs.get(q.qid, '')}".strip(), DEPTH)
        ],
        "HyDE passage alone": lambda q: (
            [d for d, _ in retrieve(index, hyde_docs[q.qid], DEPTH)]
            if hyde_docs.get(q.qid, "").strip()
            else [d for d, _ in retrieve(index, q.question, DEPTH)]
        ),
        "PRF + HyDE fused": lambda q: [
            d
            for d, _ in rrf(
                [
                    retrieve(index, prf_query(index, corpus, q.question), POOL),
                    retrieve(index, f"{q.question} {hyde_docs.get(q.qid, '')}".strip(), POOL),
                ],
                k=DEPTH,
            )
        ],
    }

    results = {}
    for name, fn in strategies.items():
        t0 = time.time()
        ranked = [(fn(q), q.gold_doc_ids) for q in queries]
        d = evaluate(ranked, seconds=time.time() - t0).as_dict()
        results[name] = d
        print(
            f"{name:26} mrr={d['mrr']:.3f}  recall@10={d['recall'][10]:.3f}  "
            f"answerable@10={d['answerable'][10]:.3f}  ({d['seconds']}s)"
        )

    out = HERE / "results.json"
    out.write_text(
        json.dumps(
            {
                "n_questions": len(queries),
                "depth": DEPTH,
                "model": MODEL,
                "empty_generations": empty,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()
