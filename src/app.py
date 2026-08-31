import os
import sys
import time
import json
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
import lightgbm as lgb
import uvicorn
from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all
from ranking_fe import build_candidate_features
from balance_relevance_constraints import filter_available, mmr_rerank, business_boost_rerank

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

STATE = {}  # populated at startup: model artifacts + reference data


def load_artifacts():
    required = ["user_embeddings.npy", "item_embeddings.npy", "restaurant_index.faiss",
                "ranker.txt", "feature_cols.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing trained model artifact(s): " + ", ".join(missing) + "\n\n"
            "Run this first:\n\n    python3 src/main_v2.py\n\n"
            f"Expected models directory: {MODELS_DIR}"
        )

    user_emb = np.load(os.path.join(MODELS_DIR, "user_embeddings.npy"))
    item_emb = np.load(os.path.join(MODELS_DIR, "item_embeddings.npy"))
    index = faiss.read_index(os.path.join(MODELS_DIR, "restaurant_index.faiss"))
    ranker = lgb.Booster(model_file=os.path.join(MODELS_DIR, "ranker.txt"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    users, restaurants, _interactions = load_all()

    return {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "index": index,
        "ranker": ranker,
        "feature_cols": feature_cols,
        "users": users,
        "restaurants": restaurants,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading trained artifacts...")
    STATE.update(load_artifacts())
    print(f"Ready: {len(STATE['users'])} users, {len(STATE['restaurants'])} restaurants loaded.")
    yield
    STATE.clear()


app = FastAPI(title="PlateMatch - Matching users to restaurants/dishes, Recommendation API (demo)", lifespan=lifespan)


@app.get("/health")
def health():
    if not STATE:
        return {"status": "not_ready"}
    return {
        "status": "ok",
        "n_users": int(len(STATE["users"])),
        "n_restaurants": int(len(STATE["restaurants"])),
        "feature_cols": STATE["feature_cols"],
    }


@app.get("/recommend")
def recommend(
    user_id: int = Query(..., ge=0, description="User id to generate recommendations for"),
    hour: int = Query(19, ge=0, le=23, description="Hour of day (0-23), used for availability + time features"),
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    diversify: bool = Query(True, description="Apply MMR diversity re-ranking"),
    boost_local: bool = Query(False, description="Boost non-chain (small/local) restaurants"),
):
    if not STATE:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if user_id >= len(STATE["user_emb"]):
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")

    timings = {}
    t_start = time.perf_counter()

    # ---------------- 1. RETRIEVAL ----------------
    t0 = time.perf_counter()
    user_vector = STATE["user_emb"][user_id].reshape(1, -1).astype(np.float32)
    n_candidates = 50
    _scores, candidate_idx = STATE["index"].search(user_vector, n_candidates)
    candidate_ids = candidate_idx[0].tolist()
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # ---------------- 2. RANKING ----------------
    t0 = time.perf_counter()
    candidate_feat = build_candidate_features(
        [user_id], [candidate_ids], STATE["users"], STATE["restaurants"], hour=hour, embedding_retriever=None
    )
    candidate_feat["rank_score"] = STATE["ranker"].predict(candidate_feat[STATE["feature_cols"]])
    timings["ranking_ms"] = (time.perf_counter() - t0) * 1000

    # ---------------- 3. CONSTRAINTS ----------------
    t0 = time.perf_counter()
    available = filter_available(candidate_feat, hour=hour)
    if diversify:
        final = mmr_rerank(available, score_col="rank_score", cuisine_col="cuisine", k=k, lambda_param=0.5)
        final["final_score"] = final["rank_score"]
    elif boost_local:
        final = business_boost_rerank(available, score_col="rank_score").head(k)
        final["final_score"] = final["boosted_score"]
    else:
        final = available.sort_values("rank_score", ascending=False).head(k)
        final["final_score"] = final["rank_score"]
    timings["constraints_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = (time.perf_counter() - t_start) * 1000

    results = [
        {
            "restaurant_id": int(row["restaurant_id"]),
            "cuisine": row["cuisine"],
            "rating": round(float(row["rating"]), 2),
            "price_tier": int(row["price_tier"]),
            "is_chain": bool(row["is_chain"]),
            "score": round(float(row["final_score"]), 4),
        }
        for _, row in final.iterrows()
    ]

    return {
        "user_id": user_id,
        "hour": hour,
        "n_candidates_retrieved": len(candidate_ids),
        "n_candidates_available": len(available),
        "recommendations": results,
        "latency_ms": {k_: round(v, 2) for k_, v in timings.items()},
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
