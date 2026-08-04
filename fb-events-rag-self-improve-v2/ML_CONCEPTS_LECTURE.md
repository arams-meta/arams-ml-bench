# A Deep Lecture on the ML in Your RAG Task

> Built around the concepts you actually used in `fb-events-rag-self-improve`.
> Line references point to your real code in `solution/solve.sh` and `instruction.md`.

---

## 0. The big picture: what problem are you solving?

You have **20,000 events** and a **query** like *"jazz shows near me this weekend"*. You must return the 10 events a person would actually want. This is the **information retrieval (IR)** problem, and it's the "R" in RAG.

There are two fundamentally different ways to match a query to documents:

| Approach | How it matches | Weakness |
|---|---|---|
| **Keyword / lexical** (e.g. BM25) | Exact word overlap: query word "jazz" must appear in the event | "jazz" ≠ "bebop quartet" — misses synonyms |
| **Semantic / embedding** | Match by *meaning*, not exact words | Needs a model; can match things that are topically close but wrong on hard facts (wrong city, wrong date) |

Your instruction (`instruction.md:1`) says *"both by keywords and embeddings"* + *"hard filters"*. That combination — **hard filters + semantic ranking** — is the heart of every real search system.

---

## 1. Embeddings — the single most important idea

> An **embedding** is a list of numbers (a *vector*) that represents the *meaning* of a piece of text, such that similar meanings land at nearby points in space.

Your model, `all-MiniLM-L6-v2`, turns any sentence into **384 numbers** (`instruction.md:26`). Think of each text as a point in a 384-dimensional space. You can't picture 384 dimensions, so picture 2D:

```
        art •  • painting

  jazz •  • blues

                       • basketball
              • soccer •
```

"jazz" and "blues" sit close; "soccer" and "basketball" sit close; the music cluster is far from the sports cluster. The model learned this from reading enormous amounts of text — words used in similar contexts get similar vectors. This is the payoff: **you can now measure "is this event about the same thing as this query?" as a distance between two points.**

In your code, this one line does all the heavy lifting (`solve.sh:81`):
```python
corpus_emb = model.encode(corpus_texts, ..., normalize_embeddings=True)
```
`corpus_emb` is a 20000 × 384 matrix — every event as a point in space.

### Why you concatenated fields
Line 78:
```python
corpus_texts=[f"{e['title']} {e['description']} {e['category']} {e['city']} {e['venue']} {e.get('topic','')}" for e in valid_events]
```
The model embeds *whatever text you give it*. By stuffing title + description + **topic** into one string, you're telling the model "consider all of this when deciding meaning." Notice you deliberately included `topic` — that's the *facet* signal the instruction says the baseline ignores and fails on (`instruction.md:11`). This is **feature engineering**: choosing what information the model sees.

---

## 2. Measuring similarity: cosine / inner product

Once query and event are both points, how "close" are they? The standard measure is **cosine similarity** — the cosine of the angle between the two vectors. It's +1 when they point the same direction (identical meaning), 0 when unrelated, −1 when opposite.

Here's the elegant trick you used. If every vector is **normalized to length 1** (that's what `normalize_embeddings=True` does), then cosine similarity equals the plain **dot product**. That's why you could write similarity as a simple matrix multiply (`solve.sh:155`):
```python
scores = (f_emb @ q_embs[i]).squeeze()   # @ is dot product
```
and why your FAISS index is `IndexFlatIP` — **IP = Inner Product** (`solve.sh:85`). Normalize first, and inner product *is* cosine. That's a deliberate, standard optimization.

---

## 3. Vector search & FAISS — finding neighbors fast

Given a query point, you want the nearest event points. Naively you compute the query's similarity to all 20,000 events and sort. That's **brute force** — and you actually wrote both paths (`solve.sh:132-139`):
```python
if use_faiss:
    scores, idx = index.search(q_emb..., 50)   # fast
else:
    scores = (corpus_emb @ q_emb)              # brute force
    top = np.argsort(-scores)[:50]
```

**FAISS** (Facebook AI Similarity Search) is a library that builds an **index** over your vectors so nearest-neighbor lookup is fast even at millions of vectors. At 20k, brute force is fine; at 20M it's not. This is the concept of **Approximate Nearest Neighbor (ANN) search** — the backbone of every vector database (Pinecone, Weaviate, etc.). `IndexFlatIP` is actually the *exact* (non-approximate) flat index; the "approximate" flavors trade a little accuracy for huge speed.

---

## 4. The killer architecture: filter-first, then rank

This is the most important design decision in your whole solution, and it's how real search works.

Some constraints are **non-negotiable facts** (hard filters):
- event must be in the query's **city**
- within the **time window**
- within **radius_km** (geographic distance)
- not spam

Others are **soft preferences** (ranking): *how well does the meaning match?*

Semantic similarity alone would happily return a perfectly on-topic jazz show — **in the wrong city, last year.** So your `get_filtered` (`solve.sh:98-117`) first throws away everything that violates a hard fact, *then* ranks the survivors by meaning. This is **filter-first / retrieve-then-rank**.

### Haversine — the geo filter
Line 18–23 is the **haversine formula**: the great-circle distance between two lat/lng points on a sphere. It's not ML — it's geometry — but it's exactly the kind of *hard constraint* that ML ranking must respect. Never let a model "decide" that Paris is close to Tokyo.

---

## 5. Re-ranking — blending multiple signals

After filtering, you don't rank on meaning *alone*. Look at `solve.sh:163`:
```python
combined = sc + ev.get("popularity",0)/100.0*0.2
```
Final score = **semantic similarity** + **0.2 × (popularity/100)**. You're mixing two signals: "is it relevant?" and "is it popular?" The `0.2` is a **weight / hyperparameter** — a knob you tune. Too high and popularity drowns out relevance; too low and it does nothing.

This is **re-ranking**: take candidates from a first-stage retriever and reorder them by a richer scoring function. Industrial systems stack many such signals (recency, click-through rate, personalization).

---

## 6. Evaluation: Recall@10, baseline, and lift

You can't improve what you can't measure. Your metric is **recall@10** (`solve.sh:119-125`):

> **Recall@k** = of all the truly relevant events, what fraction did I catch in my top *k*?

```python
rec.append(len(set(r_ids[:k]) & set(rel)) / len(rel))
```
`set(retrieved) & set(relevant)` is the intersection (how many I got right); divide by total relevant. Recall answers *"did I miss anything?"* — the right question for search, where missing a great event is worse than including a mediocre one. (Its sibling, **precision**, asks *"of what I returned, how much was junk?"*)

Two more evaluation ideas you used:

- **Baseline** (`solve.sh:127-141`): a deliberately dumb method (global semantic search, no filters) to prove your smart method actually adds value. Instruction pins it at ~0.25 recall (`instruction.md:36`).
- **Lift** (`solve.sh:273`): `improved − baseline`. Improvement in *absolute terms*. Your task demands lift > 0.08 — you must beat the dumb baseline by a real margin, not just clear an absolute bar.

**Why the grader recomputes recall** (`instruction.md:31`): self-reported scores can be gamed or buggy. Trusting only recomputed-from-outputs numbers is a core principle of honest ML evaluation.

---

## 7. Generalization & the holdout — the deepest idea here

This is the concept that separates ML from mere data-fitting.

Your holdout queries are **shifted +35 days** into the future (`instruction.md:6`). Why? Because the real test of a model isn't "does it work on the data I tuned on?" but **"does it work on data it has never seen?"**

- If your method only works on the 500 training queries → you **overfit** (memorized quirks).
- If it works nearly as well on the 100 unseen, time-shifted holdout queries → it **generalizes** (learned the real signal).

Your gate (`instruction.md:37`): `holdout_recall / train_recall >= 0.8`. A ≥20% drop means overfitting. This train/holdout(test) split is *the* foundational discipline of ML. The time-shift makes it a **distribution-shift** test — the future looks a bit different from the past, and a good system survives that.

---

## 8. Deduplication & clustering

Events with the same `cluster_id` are near-duplicates (`instruction.md:4`). Returning 5 copies of the same concert wastes your 10 slots. Your `dedup_ids` (`solve.sh:25-38`) keeps only the highest-popularity event per cluster. The idea that "similar items form clusters, collapse them to representatives" is **clustering** — one of the classic unsupervised-learning families. Here it's given to you as labels, but the intuition is the same.

---

## 9. The self-improvement loop & query rewriting

Your instruction frames this as a "self-improvement loop" (`instruction.md:39`), and your code does a simple, real version of it (`solve.sh:171-199`):

1. Run the improved retriever.
2. **Find the queries where recall < 0.5** — the failures.
3. For *those*, try a different technique — **query rewriting / query expansion**:
   ```python
   exp_texts=[f"{q['text']} . Topic {q.get('facet','')} . Category {q.get('category') or ''}" for q in low_qs]
   ```
   You *rewrite the query* by appending the topic/category, giving the embedding model more signal, then re-retrieve and merge.

This is the essence of **error analysis → targeted iteration**: don't tune blindly; look at what's failing and fix that specifically. Query rewriting/expansion is a whole subfield (HyDE, multi-query, etc.), and you built its seed.

---

## 10. Determinism & reproducibility

`random.seed(42); np.random.seed(42)` (`solve.sh:13`) and `instruction.md:28`. ML has randomness (initialization, sampling, batching). Fixing the **seed** means the same code gives the same result every run — essential for debugging, fair comparison, and grading. A result you can't reproduce isn't a result.

---

## Concept map — one page to keep

```
QUERY ──► [1] embed to 384-dim vector  (MiniLM, sentence-transformers)
              │
EVENTS ─► [1] embed all 20k → matrix   (feature-engineered text: +topic)
              │
        [4] HARD FILTERS first: city, time, radius(haversine), spam
              │  (keep only factually-valid candidates)
              ▼
        [2] similarity = cosine = dot product (because normalized)
        [3] FAISS / brute force to find nearest
              ▼
        [5] RE-RANK: semantic_score + 0.2·popularity
              ▼
        [9] error analysis → rewrite low-recall queries → merge
              ▼
        [6] EVAL: recall@10 vs baseline → lift
        [7] HOLDOUT: does it generalize to +35 days?  (overfitting check)
        [8] dedup by cluster   [10] seed=42 reproducibility
```

---

## What to study next (in order, beginner-friendly)

1. **Embeddings & cosine similarity** — the sentence-transformers docs "Semantic Search" page. Play with `.encode()` on your own sentences and print similarities. Highest-leverage hour you can spend.
2. **Recall vs precision vs F1** — understand *why* each metric answers a different business question.
3. **Train/validation/test splits & overfitting** — Andrew Ng's *Machine Learning* (Coursera) or *StatQuest* on YouTube ("Machine Learning Fundamentals: Bias and Variance").
4. **Vector databases / ANN** — the FAISS wiki intro; then the concept of HNSW indexes.
5. **RAG end-to-end** — once retrieval clicks, the "G" (feeding retrieved docs to an LLM) is a short step. The original RAG paper (Lewis et al., 2020) is readable after the above.

---

## Glossary (quick reference)

- **Embedding** — vector of numbers representing meaning.
- **Cosine similarity** — angle-based similarity; equals dot product for normalized vectors.
- **FAISS / ANN** — fast nearest-neighbor search over vectors.
- **Hard filter** — a factual constraint that must hold (city, time, geo).
- **Re-ranking** — reordering candidates by a richer scoring function.
- **Recall@k** — fraction of relevant items caught in top k.
- **Baseline / lift** — a dumb reference method / how much you beat it.
- **Holdout** — unseen data used to test generalization.
- **Overfitting** — working on tuned data but failing on new data.
- **Seed** — fixed randomness for reproducibility.
