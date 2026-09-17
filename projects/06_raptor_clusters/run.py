"""06 - RAPTOR's tree, without the summariser.

RAPTOR clusters passages, asks a model to summarise each cluster, embeds the
summaries, and retrieves over passages and summaries together. The claim is that
a question whose answer is spread thin across many passages matches a summary
better than any single passage.

Two separate ideas are bundled there: the *structure* (a second retrieval level
of coarser units) and the *summariser* (a model writing those units). This builds
the structure and replaces the summariser with extraction -- a cluster's
"summary" is the sentences closest to its own centroid, chosen by cosine, written
by nobody.

If the structure carries the benefit, this shows it at zero generation cost. If
it does not, RAPTOR's gain is the summariser's prose, which is a different and
much more expensive claim.

    level 0   the 14,602 passages
    level 1   k clusters, each represented by its most central sentences
    search    both levels; a cluster hit expands to its member passages

    python run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.corpus import load  # noqa: E402
from shared.metrics import evaluate  # noqa: E402
from shared.pipeline import Index  # noqa: E402
from shared.retrieval import BM25, Dense, rrf  # noqa: E402

HERE = Path(__file__).resolve().parent
N_QUESTIONS = 1500
DEPTH = 20
POOL = 50
SEED = 17


def build_clusters(embeddings, n_clusters: int, seed: int = SEED):
    """Mini-batch k-means, written here so the repo keeps its dependency list short."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    centroids = embeddings[rng.choice(n, size=n_clusters, replace=False)].copy()

    assign = np.zeros(n, dtype=np.int32)
    for _ in range(12):
        # Embeddings are L2-normalised, so a dot product is cosine similarity.
        sims = embeddings @ centroids.T
        new_assign = sims.argmax(axis=1).astype(np.int32)
        if (new_assign == assign).all():
            break
        assign = new_assign
        for c in range(n_clusters):
            members = embeddings[assign == c]
            if len(members):
                v = members.mean(axis=0)
                norm = np.linalg.norm(v)
                centroids[c] = v / norm if norm else centroids[c]
    return assign, centroids


def extractive_summary(corpus, member_ids, embeddings, centroid, max_sentences: int = 6) -> str:
    """The cluster's own most central sentences. No model writes anything."""
    scored = [(float(embeddings[d] @ centroid), d) for d in member_ids]
    scored.sort(reverse=True)
    picked = [d for _, d in scored[:max_sentences]]
    titles = " ".join(corpus.get(d).title for d in picked)
    bodies = " ".join(corpus.get(d).sentences[0] for d in picked if corpus.get(d).sentences)
    return f"{titles}. {bodies}"


def main() -> None:
    import numpy as np

    corpus, queries = load(N_QUESTIONS)
    index = Index.build(corpus)
    embeddings = np.asarray(index.dense._emb)  # type: ignore[union-attr]
    print(f"corpus: {len(corpus)} passages, queries: {len(queries)}\n")

    def base_rank(question: str) -> list[int]:
        sparse = index.bm25.search(question, POOL)
        dense = index.dense.search(question, POOL)  # type: ignore[union-attr]
        return [d for d, _ in rrf([sparse, dense], k=DEPTH)]

    baseline_ranked = [(base_rank(q.question), q.gold_doc_ids) for q in queries]
    base = evaluate(baseline_ranked).as_dict()
    print(f"{'baseline (passages only)':32} answerable@10={base['answerable'][10]:.3f}")

    results = {"baseline": base, "levels": {}}
    for n_clusters in (200, 800, 2000):
        t0 = time.time()
        assign, centroids = build_clusters(embeddings, n_clusters)
        members: dict[int, list[int]] = {}
        for doc_id, c in enumerate(assign):
            members.setdefault(int(c), []).append(doc_id)

        summaries, cluster_ids = [], []
        for c, ids in members.items():
            summaries.append(extractive_summary(corpus, ids, embeddings, centroids[c]))
            cluster_ids.append(c)
        # RAPTOR retrieves over passages AND summaries in ONE index, rather than
        # fusing two separate rankings. That distinction decides the result: an
        # equal-weight fusion lets fifty mostly-irrelevant cluster members dilute a
        # good passage ranking, which measures the fusion rule, not the tree.
        passage_texts = [f"{p.title}. {p.text}" for p in corpus.passages]
        combined = passage_texts + summaries
        comb_bm25 = BM25().index(combined)
        comb_dense = Dense().index(combined)
        n_passages = len(passage_texts)
        build_s = time.time() - t0

        t0 = time.time()
        ranked_out = []
        for q in queries:
            hits = rrf(
                [comb_bm25.search(q.question, POOL), comb_dense.search(q.question, POOL)],
                k=POOL,
            )
            merged: list[int] = []
            for unit_id, _ in hits:
                if unit_id < n_passages:
                    if unit_id not in merged:
                        merged.append(unit_id)
                else:
                    # A summary hit stands in for its cluster, best members first.
                    c = cluster_ids[unit_id - n_passages]
                    ids = sorted(members[c], key=lambda d: -float(embeddings[d] @ centroids[c]))
                    for d in ids[:5]:
                        if d not in merged:
                            merged.append(d)
                if len(merged) >= DEPTH:
                    break
            ranked_out.append((merged[:DEPTH], q.gold_doc_ids))
        d = evaluate(ranked_out, seconds=time.time() - t0).as_dict()
        d["clusters"] = n_clusters
        d["mean_cluster_size"] = round(len(corpus) / n_clusters, 1)
        d["build_seconds"] = round(build_s, 1)
        results["levels"][str(n_clusters)] = d
        print(
            f"{'+ ' + str(n_clusters) + ' clusters':32} answerable@10={d['answerable'][10]:.3f}  "
            f"(mean {d['mean_cluster_size']} passages/cluster, build {d['build_seconds']}s)"
        )

    out = HERE / "results.json"
    out.write_text(
        json.dumps({"n_questions": len(queries), "depth": DEPTH, **results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()
