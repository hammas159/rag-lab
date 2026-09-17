<h1 align="center">rag-lab</h1>
<p align="center"><i>Eight RAG techniques measured as retrieval, on one real corpus, with almost no language model in the loop</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/projects-8-blue" alt="projects">
  <img src="https://img.shields.io/badge/corpus-14%2C602%20real%20passages-blue" alt="corpus">
  <img src="https://img.shields.io/badge/queries-1%2C500%20with%20gold%20labels-blue" alt="queries">
  <img src="https://img.shields.io/badge/LLM%20required-1%20of%208-success" alt="llm">
</p>

---

> ### Six of the eight techniques lost to a plain hybrid baseline. A method from 1971 beat HyDE. The two that won, won only where theory said they should — and the averages hid it.

Every variant here is a **retrieval** technique, so it is measured as one. No
generation scoring, no LLM-as-judge. Putting a generator at the end of the
pipeline adds its noise on top of the effect you are trying to see, which is a
large part of why published RAG comparisons disagree with each other.

Only **one of the eight** projects needs a model at all, and it is the one that
loses.

## The corpus

HotpotQA's distractor split, pooled and de-duplicated by title.

| | |
|---|---:|
| passages | **14,602** |
| tokens | 1,297,676 |
| median passage | 79 tokens |
| queries | **1,500** |
| gold passages per query | **2** (all of them) |
| difficulty | all `hard` |

The distractors are the point: each question ships with two paragraphs the
annotators used and eight plausible wrong ones from the same Wikipedia
neighbourhood. A corpus where only the right passage mentions the topic measures
nothing.

## The metric that changes the conclusions

Every question needs **both** gold passages, so this lab reports `answerable@k`
— *were all required passages retrieved* — next to recall.

```
BM25 @10 :  recall 81.3%   answerable 64.2%
```

Recall flatters it by **17 points**. A pipeline holding one of two bridging
paragraphs cannot answer, however good its recall looks. Most RAG benchmarks
report the flattering number.

## Baselines

| Strategy | MRR | recall@10 | **answerable@10** | search |
|---|---:|---:|---:|---:|
| BM25 (implemented here, no dependency) | 0.810 | 0.813 | 0.642 | 4.6s |
| Dense (`bge-small-en-v1.5`) | **0.928** | 0.891 | 0.792 | 23.4s |
| **Hybrid, RRF fused** | 0.891 | **0.905** | **0.813** | 28.7s |

⚠️ **This contradicts [nlp-lab](https://github.com/hammas159/nlp-lab)**, where BM25 landed
within 8 points of a pretrained embedding. Here dense wins by **15 points** on
answerable@10 — because HotpotQA asks paraphrased questions, not keyword
lookups. Neither result is wrong, which is the argument for measuring on your own
corpus instead of trusting a leaderboard.

---

## Input

![input](docs/images/input.png)

## Output

`python projects/03_iterative_multihop/run.py`

![output](docs/images/output.png)

---

## Results

Everything is measured against the same hybrid baseline: **answerable@10 = 0.813**.

| # | Technique | Best result | vs baseline | Needs an LLM? |
|---|---|---:|---:|:---:|
| 04 | **PRF + HyDE fused** | **0.852** | **+3.9** | partly |
| 03 | **Iterative two-hop, routed** | **0.853** | **+4.0** | no |
| 04 | PRF alone (no model) | 0.831 | +1.8 | no |
| 08 | Speculative (hybrid + verifier, pool 20) | 0.826 | +1.3 | no |
| 01 | Retrieval unit (paragraph) | 0.805 | −0.8 | no |
| 04 | HyDE alone (7b-instruct) | 0.803 | −1.0 | **yes** |
| 07 | Entity graph expansion | 0.805 | −0.8 | no |
| 05 | Corrective RAG (selective) | 0.799 | −1.4 | no |
| 06 | RAPTOR clusters | 0.791 | −2.2 | no |
| 02 | RAG-Fusion (multi-query) | 0.762 | −5.1 | no |

---

### [01 · What should a retrieval unit be?](projects/01_retrieval_unit/)

| granularity | units | mean tokens | answerable@10 |
|---|---:|---:|---:|
| title only | 14,602 | **3** | 0.415 |
| sentence | 59,784 | 25 | 0.775 |
| **paragraph** | 14,602 | 92 | **0.805** |

**Smaller is not sharper.** Sentences cost **4× the index** and lose 3 points.
And the free win nobody mentions: **prepending the title to a passage is worth
+4.0 points** (BM25 0.642 → 0.682). Title-only retrieval — three tokens, no body
text — already answers **41.5%**.

### [02 · RAG-Fusion without a language model](projects/02_rag_fusion/)

| views fused | answerable@10 | queries/question |
|---|---:|---:|
| **original only** | **0.813** | 1.00 |
| + keywords | 0.807 | 2.00 |
| + keywords + entities | 0.764 | 2.97 |
| all five views | 0.762 | 3.80 |

**Every config loses, monotonically** — 3.6× the cost to give up 5 points. The
rewrites are rule-based, which isolates the fusion and leaves a falsifiable
claim: *if RAG-Fusion helps, the LLM's paraphrase is doing the work, not the RRF.*

### [03 · Multi-hop retrieval without a reasoner](projects/03_iterative_multihop/) ⭐

| strategy | overall | bridge | comparison |
|---|---:|---:|---:|
| single hop | 0.813 | 0.769 | **0.987** |
| two hops, always | 0.846 | **0.840** | 0.870 |
| **two hops, bridge only** | **0.853** | 0.820 | 0.984 |

**+7.1 points on bridge questions with no model.** Round two's query is just the
question plus vocabulary from round one — the retrieved passage supplies the
words the question lacked.

But applied blindly it **costs 11.7 points** on comparison questions, which
already name both entities. Routing recovers most of it, and is capped by the
router: the rule classifier has recall 0.977 but **precision 0.386**, and its 477
misroutes are exactly why routed bridge (0.820) trails always-two-hop (0.840).

### [04 · HyDE versus a technique from 1971](projects/04_hyde_vs_prf/)

The only project here that uses a model — `qwen2.5:7b-instruct`, 1,500
generations, 44 minutes, zero failures.

| strategy | recall@10 | answerable@10 |
|---|---:|---:|
| baseline (question only) | 0.905 | 0.813 |
| **PRF — pseudo-relevance feedback, no model** | 0.910 | **0.831** |
| HyDE (7b-instruct) | 0.898 | **0.803** |
| HyDE passage alone | 0.881 | 0.771 |
| **PRF + HyDE fused** | **0.923** | **0.852** |

**HyDE loses to the baseline, and to a method from 1971.** Both replace the
question with something answer-shaped; only one needs a GPU.

The nuance that saves it: fusing them beats either alone. HyDE contributes
something *orthogonal* even while being unhelpful by itself.

### [05 · Corrective RAG: can retrieval detect its own failure?](projects/05_corrective_rag/)

CRAG needs an evaluator that knows retrieval went wrong. The retriever already
emits confidence signals, so the question is whether they mean anything.

| signal | when answerable | when **not** |
|---|---:|---:|
| top score | 0.0325 | 0.0325 |
| margin (rank 1 − rank 2) | 0.0009 | **0.0019** |
| BM25/dense agreement | 0.539 | **0.657** |

**All three are flat or inverted.** Margin and agreement are *higher* when
retrieval has failed. So selective correction fires on the wrong queries —
trigger precision 0.146 and 0.117 against a 0.187 base failure rate, i.e. **worse
than random** — and both selective strategies land below doing nothing.

Correcting *everything* does work (0.846), which is project 03's result. The
selection is what fails.

### [06 · RAPTOR's tree, without the summariser](projects/06_raptor_clusters/)

Clusters built by k-means; each represented by its own most central sentences
rather than a model's prose.

| level | answerable@10 |
|---|---:|
| baseline, passages only | **0.813** |
| + 200 clusters | 0.791 |
| + 800 clusters | 0.784 |
| + 2,000 clusters | 0.786 |

**The structure alone costs 2–3 points.** Either the summariser is doing the
work, or HotpotQA is the wrong corpus for it — RAPTOR targets questions needing
broad thematic synthesis, and these are precise two-hop factoid lookups. **That
caveat is real; this is not a refutation of RAPTOR in general.**

🛠 **I got this wrong twice before getting it right**, and both wrong versions
are recorded in the file. Appending cluster hits *after* 50 passage hits at depth
20 made them unreachable and produced a fake "identical to baseline". Fusing them
equal-weight produced a fake **0.406**. Only summaries in the *same* index — what
RAPTOR actually does — is a fair test.

### [07 · GraphRAG's graph, built by a regular expression](projects/07_entity_graph/)

96,113 entities, 15,243 linking, 196,691 edges, built in **1.0 second** with no
model. Hub entities over 40 passages are dropped — the graph equivalent of an idf
floor.

| strategy | overall | bridge | comparison |
|---|---:|---:|---:|
| baseline (text only) | **0.813** | 0.769 | **0.987** |
| graph expansion | 0.805 | **0.777** | 0.912 |

It helps **exactly where theory says it should** — bridge questions, +0.8 — and
hurts comparison by 7.5. But iterative retrieval (project 03) gets **9× the
bridge gain** with less machinery.

⚠️ Rule-based entity extraction is crude. A model-built graph may do better; this
measures the structure, not the ceiling.

### [08 · Speculative retrieval: how weak may the drafter be?](projects/08_speculative_rerank/)

A cross-encoder cannot scan a corpus — it can only reorder what something else
proposed. So the question is how bad that proposal may be.

| drafter | alone | pool 20 | pool 50 | pool 100 |
|---|---:|---:|---:|---:|
| BM25 | 0.608 | 0.756 | 0.800 | **0.810** |
| dense | 0.798 | 0.822 | **0.826** | 0.826 |
| hybrid | **0.824** | **0.826** | 0.824 | 0.820 |

**The verifier's value is inversely proportional to the drafter's quality**: it
lifts BM25 by **20.2 points** and hybrid by **0.2**.

And for a good drafter, a **bigger pool is worse** — hybrid goes 0.826 → 0.824 →
0.820 as the pool grows. More candidates give the cross-encoder more chances to
promote a distractor. The pool size is not a quality knob you turn up.

A cheap drafter can be rescued to near-hybrid quality, at **492 ms/query against
31 ms**. Building the dense index once (47s) is cheaper than paying a
cross-encoder on every query to recover from a weak one.

---

## What the eight say together

**RRF is not free.** It dilutes when one input is weak (02's degraded rewrites,
06's cluster members) and helps when both are strong (04's PRF + HyDE). Every
fusion result here is explained by that one rule.

**Averages hide the only real wins.** Iterative retrieval and graph expansion
both help bridge questions and hurt comparison questions. Reported as a single
number, project 03 looks like +3.3; split by type it is +7.1 and −11.7, and the
routing decision follows immediately.

**Cheap and old beats expensive and new, twice.** PRF (1971) beats HyDE. A
regular expression builds a usable entity graph in one second.

## Running it

```bash
python projects/01_retrieval_unit/run.py        # ... through 08
```

The corpus is sampled with a fixed seed and cached, so every project measures
byte-identical data. A variant that looked better on a different sample would not
be a finding.

## Layout

```
shared/corpus.py      HotpotQA -> de-duplicated passages + gold labels, cached
shared/retrieval.py   BM25 (own, inverted index), dense, RRF, cross-encoder
shared/metrics.py     recall@k, answerable@k, nDCG@k, MRR
shared/pipeline.py    the baselines every variant is measured against
projects/01..08/      one technique each, each writing results.json
```

## Stack

`Python 3.11+` · `sentence-transformers` (`bge-small-en-v1.5`,
`ms-marco-MiniLM-L-6-v2`) · `datasets` · `numpy` · `pytest` · `ollama`
(`qwen2.5:7b-instruct`, project 04 only) — **BM25 and k-means are implemented
here**, not imported.

## Not measured

Self-RAG and FLARE, which both need token-level model internals (reflection
tokens, mid-generation logprobs) rather than a pipeline change. Everything else
in the RAG-variant literature that could be approximated without a generator is
above.

## Licence

MIT — see [LICENSE](LICENSE).
