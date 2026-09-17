"""07 - GraphRAG's graph, built by a regular expression.

GraphRAG asks a model to read every passage, name the entities, and link them.
That is the expensive half. The cheap observation is that a bridge question is
already a graph query: the first passage names something the second passage is
about, and the question names neither connection explicitly.

Entities here are extracted by a rule -- capitalised spans, plus the passage
title, which in a Wikipedia corpus is the entity. Two passages share an edge when
they mention the same entity. Retrieval seeds with ordinary text search and then
walks one hop along those edges.

No model reads anything. If the graph structure is what makes GraphRAG work, it
should show up here, and it should show up **on bridge questions specifically** --
which is the part an averaged number would hide.

    python run.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.corpus import load  # noqa: E402
from shared.metrics import evaluate  # noqa: E402
from shared.pipeline import Index  # noqa: E402
from shared.retrieval import rrf  # noqa: E402

HERE = Path(__file__).resolve().parent
N_QUESTIONS = 1500
DEPTH = 20
POOL = 50

_ENTITY = re.compile(
    r"\b[A-Z][\w'’&-]*(?:\s+(?:of|the|de|van|der|and)\s+[A-Z][\w'’&-]*|\s+[A-Z][\w'’&-]*)*"
)


def entities(text: str, *, min_len: int = 4) -> set[str]:
    """Capitalised spans. Crude on purpose -- the point is that it needs no model."""
    found = set()
    for m in _ENTITY.finditer(text):
        span = m.group(0).strip(" .,;:'’")
        if len(span) >= min_len and not span.isupper():
            found.add(span.lower())
    return found


def build_graph(corpus, *, max_entity_docs: int = 40):
    """entity -> passages, and passage -> neighbouring passages.

    An entity mentioned by hundreds of passages ("United States") is a hub that
    connects everything to everything; it carries no information and would make
    every passage a neighbour of every other. Those are dropped, which is the
    graph equivalent of an idf floor.
    """
    ent_to_docs: dict[str, set[int]] = defaultdict(set)
    for p in corpus.passages:
        ents = entities(p.text) | {p.title.lower()}
        for e in ents:
            ent_to_docs[e].add(p.doc_id)

    dropped = {e for e, docs in ent_to_docs.items() if len(docs) > max_entity_docs}
    kept = {e: d for e, d in ent_to_docs.items() if 1 < len(d) <= max_entity_docs}

    neighbours: dict[int, Counter] = defaultdict(Counter)
    for _e, docs in kept.items():
        docs_list = sorted(docs)
        for a in docs_list:
            for b in docs_list:
                if a != b:
                    neighbours[a][b] += 1
    return kept, neighbours, len(ent_to_docs), len(dropped)


def main() -> None:
    corpus, queries = load(N_QUESTIONS)
    index = Index.build(corpus)
    print(f"corpus: {len(corpus)} passages, queries: {len(queries)}")

    t0 = time.time()
    ent_to_docs, neighbours, n_entities, n_dropped = build_graph(corpus)
    edges = sum(len(v) for v in neighbours.values()) // 2
    print(
        f"graph: {n_entities} entities ({n_dropped} hubs dropped), "
        f"{len(ent_to_docs)} linking entities, {edges} edges, built in {time.time() - t0:.1f}s\n"
    )

    def text_hits(question: str, k: int):
        return rrf(
            [index.bm25.search(question, POOL), index.dense.search(question, POOL)],  # type: ignore[union-attr]
            k=k,
        )

    def graph_expand(question: str, seeds: int = 3, per_seed: int = 6) -> list[int]:
        hits = text_hits(question, POOL)
        base = [d for d, _ in hits]
        expanded: list[int] = []
        for d in base[:seeds]:
            for nb, _weight in neighbours[d].most_common(per_seed):
                if nb not in base[:seeds] and nb not in expanded:
                    expanded.append(nb)
        merged = list(base[:seeds])
        # Interleave: a neighbour of the best passage is worth more than the
        # twentieth text hit, but not more than the second text hit.
        rest = [d for d in base[seeds:] if d not in expanded]
        for i in range(max(len(expanded), len(rest))):
            if i < len(expanded) and expanded[i] not in merged:
                merged.append(expanded[i])
            if i < len(rest) and rest[i] not in merged:
                merged.append(rest[i])
        return merged[:DEPTH]

    strategies = {
        "baseline (text only)": lambda q: [d for d, _ in text_hits(q.question, DEPTH)],
        "graph expansion": lambda q: graph_expand(q.question),
    }

    results = {
        "graph": {
            "entities_total": n_entities,
            "hubs_dropped": n_dropped,
            "linking_entities": len(ent_to_docs),
            "edges": edges,
        },
        "strategies": {},
    }
    for name, fn in strategies.items():
        t0 = time.time()
        ranked = [(fn(q), q.gold_doc_ids) for q in queries]
        overall = evaluate(ranked, seconds=time.time() - t0).as_dict()
        by_type = {}
        for hop_type in ("bridge", "comparison"):
            subset = [
                (r, g) for (r, g), q in zip(ranked, queries, strict=False) if q.hop_type == hop_type
            ]
            by_type[hop_type] = evaluate(subset).as_dict()
        results["strategies"][name] = {"overall": overall, "by_type": by_type}
        print(
            f"{name:22} answerable@10: overall={overall['answerable'][10]:.3f}  "
            f"bridge={by_type['bridge']['answerable'][10]:.3f}  "
            f"comparison={by_type['comparison']['answerable'][10]:.3f}  ({overall['seconds']}s)"
        )

    out = HERE / "results.json"
    out.write_text(
        json.dumps({"n_questions": len(queries), "depth": DEPTH, **results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()
