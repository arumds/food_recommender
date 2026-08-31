# Personalized Restaurant Discovery Pipeline

A runnable, end-to-end recommendation system demo covering **retrieval,
ranking, personalization, and product-constraint handling** — the core
concepts for building recommendation systems.

It uses **synthetic data** (no real data), generated with realistic
structure: users have latent cuisine preferences, restaurants have popularity
and availability windows, and interactions are simulated with a separate
*exposure* model (what gets shown) and *outcome* model (does it convert to an
order) — this mirrors the real logging-policy-vs-outcome distinction you deal
with in production recommender systems.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Data generation and the recommender pipeline are two **independent steps**.
`data_gen.py` only ever writes to `data/*.csv` and does nothing else;
`main.py` only ever reads from `data/*.csv` via `data_loader.py` and never
generates data itself. This mirrors a real system, where a data/ETL job and
the model pipeline that consumes its output are separate,
independently-schedulable processes — and it means every pipeline run reads
the exact same dataset, so you can fairly compare model changes across runs
instead of comparing against a freshly-randomized dataset each time.

```bash
# Step 1: generate the dataset (run once, or re-run any time you want a fresh dataset)
python3 src/data_gen.py

# Step 2: run the pipeline against whatever is currently in data/
python3 src/main.py
```

If you skip Step 1, Step 2 fails immediately with a clear message telling you
to run `data_gen.py` first, rather than silently generating data inline.

Running the pipeline takes about 1-2 minutes (it trains a two-tower neural
net, builds a FAISS index, trains LambdaMART, and runs an 8000-round bandit
simulation), prints a full report to stdout, and saves
`outputs/results_summary.json`.

## What the pipeline covers

One script, `main.py`, runs every stage in order:

| Stage | Objective | Model(s) used |
|---|---|---|
| **Retrieval** | Fast candidate generation from the full restaurant catalog | `two_tower.py` (PyTorch two-tower NN, consumes user affinity + restaurant features + lat/lon) + `faiss_index.py` (FAISS HNSW approximate NN index) |
| **Ranking** | Precisely score the retrieved candidates | `ranking.py` (LightGBM LambdaMART, pairwise/listwise learning-to-rank) |
| **Constraints** | Adjust the ranked list for product/business needs the ranker doesn't see | `constraints.py` — hard availability filtering, MMR diversity re-ranking, business-rule boosting for small/local restaurants |
| **Personalization** | Real-time context-aware re-ranking | `bandit.py` (LinUCB contextual bandit, adapts online to weather/time-of-day/discount context) |
| **Online evaluation** | Estimate a launch decision before running a real A/B test | `evaluate.py` — a toy Inverse Propensity Scoring (IPS) estimator |

Every stage prints its own metrics and the full run is summarized into JSON
at the end. **It also saves the trained retrieval (Two-Tower + FAISS) and
ranking (LambdaMART) artifacts to `models/`** — the same run that evaluates
the architecture produces what `serve.py` needs to serve it live, so there's
no separate training step to remember to run.

## Serving this online (minimal demo)

`main.py` is a batch/offline script — it evaluates the architecture and
saves model artifacts in one run, but doesn't expose anything as a live
service itself. `app.py` is a separate, minimal demonstration of what 
serving those artifacts for real would look like:

```bash
python3 src/data_gen.py    # if you haven't already
python3 src/main.py     # evaluates everything AND saves models/
python3 src/app.py       # starts a FastAPI server on http://127.0.0.1:8000       
```

Then, in another terminal:

```bash
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/recommend?user_id=0&hour=19&k=10"
```

Or open `http://127.0.0.1:8000/docs` for FastAPI's interactive API explorer.

**Why `app.py` is still a separate script, even though training and
evaluation are merged**: `app.py` only ever *loads* what `main.py`
already produced and answers requests against it — same design principle as
`data_loader.py` only reading what `data_gen.py` wrote. A live service
shouldn't be retraining a two-tower network and a LambdaMART model on every
request; it loads a fixed artifact and serves against it, which is exactly
what separating these two processes forces you to get right.

**What each request actually does** (and roughly how long each stage takes,
measured on this ~1500-user/400-restaurant dataset after the FAISS index has
warmed up):

| Stage | What happens | Typical latency |
|---|---|---|
| Retrieval | Look up the user's precomputed embedding, FAISS HNSW search → ~50 candidates | <1ms |
| Ranking | Build features for those candidates, score with the loaded LambdaMART model | ~5-10ms |
| Constraints | Filter closed restaurants, optionally MMR-diversify or business-boost | ~3-5ms |
| **Total** | | **~10-15ms** |

**Deliberately excluded from `app.py`**: the LinUCB bandit. It's
*stateful* — its per-restaurant parameters need to persist and update
between requests (normally via a shared store like Redis), which is real
infrastructure beyond what a "minimal" demo is meant to show. It still runs
and gets evaluated inside `main.py` — it's just not part of the servable
API.

## Project structure

```
src/
  data_gen.py       # STANDALONE data-generation script. Only writes data/*.csv; never imported by the pipeline for generation.
  data_loader.py    # Loads data/*.csv for the pipeline. Only reads; never generates. Fails loudly if data_gen.py hasn't run yet.
  two_tower.py      # PyTorch Two-Tower NN retriever (consumes user/restaurant side features)
  faiss_index.py    # FAISS FlatIP (exact) + HNSW (approximate) ANN indexes
  ranking.py        # LightGBM LambdaMART learning-to-rank + feature engineering
  constraints.py    # availability filtering, MMR diversity re-ranking, business-rule boosting
  context.py        # synthetic real-time context (weather/time-of-day/discount) + reward model
  bandit.py         # LinUCB contextual bandit re-ranker
  evaluate.py       # Recall@K, NDCG@K, MAP@K, + a toy IPS online-lift estimator
  main.py        # the pipeline runner -- evaluates every stage AND saves models/ artifacts
  app.py          # FastAPI app -- loads models/ artifacts, serves /recommend and /health
data/            # generated CSVs (users, restaurants, interactions) -- created by data_gen.py
models/          # trained artifacts (embeddings, FAISS index, ranker) -- created by main.py
outputs/         # results_summary.json from the last run
```

## Git history as a build log

Each component was implemented, tested against a baseline, and committed
independently — the commit log itself is a useful walkthrough of how this
was built, including additions that were later removed once they'd served
their purpose:

```bash
git log --oneline
```

Run `git show <hash>` on any commit to see that component's diff, or
`git log -p -- src/two_tower.py` (etc.) to see a single file's history.

## Mapping the concepts to the code

| Concept | Where it lives |
|---|---|
| **Retrieval** | `two_tower.py` + `faiss_index.py` — a real two-tower neural network with an approximate-NN index, benchmarked against exact search for recall and latency. |
| **Ranking** | `ranking.py` — LightGBM LambdaMART (pairwise/listwise learning-to-rank), the same model family used in production ranking systems, evaluated with NDCG@K / MAP@K. |
| **Personalization** | `bandit.py` (LinUCB real-time contextual re-ranking) — adapts online to weather/time-of-day/discount context, unlike the offline-trained ranker which never adapts after training. |
| **Diversity / availability / business constraints** | `constraints.py` — hard availability filtering, MMR re-ranking (classic IR diversification), and a "boost small/local restaurants" business rule, each with before/after metrics so the tradeoff is visible, not just asserted. |
| **Offline evaluation rigor** | Time-based (not random) train/test split to avoid leakage; Recall@K / NDCG@K / MAP@K implemented from scratch in `evaluate.py`. |
| **Online experiment interpretation** | A toy Inverse Propensity Scoring (IPS) estimator, illustrating the offline-estimate-before-you-ship technique — explicitly flagged as **not** a substitute for a real A/B test. |

## Honest caveats

- **This is synthetic data with a hand-built generative process** — it's built to have learnable structure, not to prove any specific algorithm is "best." Real data would have much messier, sparser, and non-stationary signal.
- **Sample sizes here are small** (1500 users, 400 restaurants), so metric differences between methods can be noisy run-to-run — in a real setting you'd want confidence intervals or a proper significance test before concluding one approach beats another.
- **The IPS online-lift estimate is illustrative only.** Real causal estimates need correctly-logged propensities from the actual serving policy, and should still be validated with a live experiment before a launch decision.
- **The bandit simulation uses a small restaurant pool** (60 of the 400 restaurants) so it converges within a reasonable number of simulated rounds — a real deployment would need either far more rounds or a smarter exploration strategy at full catalog scale.

## Extending this further

- **Add back Multi-gate Mixture-of-Experts (MMoE) for multi-objective personalization.**
  This was in an earlier version of the pipeline (see git history) and was
  removed to keep the personalization story focused on one technique, but
  it addresses a real gap: `bandit.py` optimizes a single reward signal,
  while a production system usually cares about several correlated-but-distinct
  objectives at once — e.g. click-through *and* order conversion *and*
  longer-term retention. MMoE handles this by running shared "expert"
  sub-networks with a separate gating network per task, so each objective
  can weight the shared representation differently instead of forcing one
  model to compromise between goals it wasn't designed to balance.
- Swap in a real public dataset (Yelp Open Dataset, Instacart, H&M) instead of synthetic data.
- Add a sequence-aware model (e.g. a simple GRU/SASRec-style session encoder) to capture *intent within a session*, not just static affinity.
- Add basic monitoring/dashboarding around `app.py` (request latency percentiles, feature drift, model staleness alerts) to speak to the "production deployment and monitoring" part of the JD.
