import os
import sys
import json
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all
from ranking_fe import build_ranking_features, build_candidate_features, time_split, train_ranker
from two_tower import train_two_tower, TwoTowerRetriever
from faiss_index import HNSWRetriever
from balance_relevance_constraints import filter_available, mmr_rerank, business_boost_rerank, intra_list_diversity, chain_share
from context import CONTEXT_VECTOR_DIM
from bandit import simulate_bandit_vs_static
from evaluate import evaluate_users

RESULTS = {}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)

    section("DATA")
    users, restaurants, interactions = load_all()
    train_raw, test_raw = time_split(interactions, test_frac=0.2)
    test_positive = test_raw[test_raw["ordered"] == 1]
    user_relevant = test_positive.groupby("user_id")["restaurant_id"].apply(list).to_dict()
    eval_users_list = list(user_relevant.keys())[:300]
    print(f"Loaded from disk: users={len(users)} restaurants={len(restaurants)} interactions={len(interactions)}")

    # =================================================================
    section("RETRIEVAL: Two-Tower NN + FAISS HNSW -- behavioral candidate generation")
    # =================================================================
    model, user_feat, item_feat = train_two_tower(train_raw, users, restaurants, epochs=8)
    tt = TwoTowerRetriever(model, user_feat, item_feat)
    hnsw_index = HNSWRetriever(tt.item_emb)

    tt_recs = {uid: hnsw_index.retrieve(tt.user_emb[uid], k=20) for uid in eval_users_list}
    tt_metrics = evaluate_users(tt_recs, user_relevant, k=20)
    print(f"Two-Tower + FAISS HNSW retrieval: {tt_metrics}")
    RESULTS["retrieval_two_tower_hnsw"] = tt_metrics

    # =================================================================
    section("RANKING: LightGBM LambdaMART over merged candidate pool")
    # =================================================================
    train_feat, feature_cols = build_ranking_features(train_raw, users, restaurants, embedding_retriever=None)
    ranker = train_ranker(train_feat, feature_cols)

    eval_candidate_lists = [hnsw_index.retrieve(tt.user_emb[uid], k=50) for uid in eval_users_list]
    candidate_feat = build_candidate_features(
        eval_users_list, eval_candidate_lists, users, restaurants, hour=19, embedding_retriever=None
    )
    candidate_feat["rank_score"] = ranker.predict(candidate_feat[feature_cols])
    ranked_recs = (
        candidate_feat.sort_values(["user_id", "rank_score"], ascending=[True, False])
        .groupby("user_id")["restaurant_id"].apply(list).to_dict()
    )
    ranker_metrics = evaluate_users(ranked_recs, user_relevant, k=10)
    print(f"LambdaMART ranker over Two-Tower candidates: {ranker_metrics}")
    RESULTS["ranking_lambdamart"] = ranker_metrics

    retrieval_order_recs = (
        candidate_feat.sort_values(["user_id", "retrieval_rank"], ascending=[True, True])
        .groupby("user_id")["restaurant_id"].apply(list).to_dict()
    )
    retrieval_order_metrics = evaluate_users(retrieval_order_recs, user_relevant, k=10)
    print(f"Retrieval order truncated to top-10 (no re-ranking): {retrieval_order_metrics}")
    RESULTS["retrieval_order_top10_baseline"] = retrieval_order_metrics

    ranking_lift_pct = (
        (ranker_metrics["Recall@10"] - retrieval_order_metrics["Recall@10"])
        / retrieval_order_metrics["Recall@10"] * 100
        if retrieval_order_metrics["Recall@10"] else None
    )
    print(f"LambdaMART re-ranking lift over retrieval-order baseline, same k=10, same pool: "
          f"{ranking_lift_pct:.1f}%" if ranking_lift_pct is not None else "n/a")
    RESULTS["ranking_lift_over_retrieval_order_pct"] = ranking_lift_pct

    # =================================================================
    section("SAVING MODEL ARTIFACTS (for app.py)")
    # =================================================================
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(MODELS_DIR, "user_embeddings.npy"), tt.user_emb.astype(np.float32))
    np.save(os.path.join(MODELS_DIR, "item_embeddings.npy"), tt.item_emb.astype(np.float32))
    faiss.write_index(hnsw_index.index, os.path.join(MODELS_DIR, "restaurant_index.faiss"))
    ranker.save_model(os.path.join(MODELS_DIR, "ranker.txt"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved retrieval + ranking artifacts to: {MODELS_DIR}")
    print("  user_embeddings.npy, item_embeddings.npy, restaurant_index.faiss, "
          "ranker.txt, feature_cols.json")

    # =================================================================
    section("CONSTRAINTS: availability, diversity (MMR), business boost")
    # =================================================================
    sample_uid = eval_users_list[0]
    sample_candidates = candidate_feat[candidate_feat["user_id"] == sample_uid].copy()

    available = filter_available(sample_candidates, hour=19)
    print(f"Candidates before availability filter: {len(sample_candidates)}, after: {len(available)}")

    top10_raw = available.sort_values("rank_score", ascending=False).head(10)
    top10_mmr = mmr_rerank(available, score_col="rank_score", cuisine_col="cuisine", k=10, lambda_param=0.5)
    top10_boosted = business_boost_rerank(available, score_col="rank_score").head(10)

    diversity_raw = intra_list_diversity(top10_raw)
    diversity_mmr = intra_list_diversity(top10_mmr)
    chain_raw = chain_share(top10_raw)
    chain_boosted = chain_share(top10_boosted)

    print(f"Diversity (unique cuisine ratio) -- raw top10: {diversity_raw:.2f}, MMR top10: {diversity_mmr:.2f}")
    print(f"Chain share -- raw top10: {chain_raw:.2f}, boosted top10: {chain_boosted:.2f}")
    RESULTS["constraints"] = {
        "candidates_before_availability_filter": int(len(sample_candidates)),
        "candidates_after_availability_filter": int(len(available)),
        "diversity_raw": diversity_raw,
        "diversity_mmr": diversity_mmr,
        "chain_share_raw": chain_raw,
        "chain_share_boosted": chain_boosted,
    }

    # =================================================================
    section("PERSONALIZATION: LinUCB contextual bandit re-ranking")
    # =================================================================
    small_restaurants = restaurants.sample(60, random_state=1).reset_index(drop=True)
    bandit_result = simulate_bandit_vs_static(
        n_rounds=8000, restaurants=small_restaurants, context_dim=CONTEXT_VECTOR_DIM, rng=rng,
    )
    print(f"Bandit mean reward, last 20% of rounds: {bandit_result['bandit_mean_reward_last_20pct']:.3f}")
    print(f"Static (context-blind) baseline, last 20% of rounds: {bandit_result['static_mean_reward_last_20pct']:.3f}")
    print(f"Bandit total reward: {bandit_result['bandit_cumulative_reward'][-1]:.0f}  "
          f"vs static total reward: {bandit_result['static_cumulative_reward'][-1]:.0f}")
    RESULTS["contextual_bandit"] = {
        "bandit_mean_reward_last_20pct": bandit_result["bandit_mean_reward_last_20pct"],
        "static_mean_reward_last_20pct": bandit_result["static_mean_reward_last_20pct"],
        "bandit_total_reward": float(bandit_result["bandit_cumulative_reward"][-1]),
        "static_total_reward": float(bandit_result["static_cumulative_reward"][-1]),
    }

    # =================================================================
    section("SIMULATED ONLINE EXPERIMENT (toy IPS estimator)")
    # =================================================================
    n_sim = 5000
    propensity = rng.uniform(0.05, 0.5, n_sim)          # P(control policy would show this item)
    treatment_relevance = rng.beta(2, 3, n_sim)          # how relevant the treatment model thinks the item is
    true_ctr_fn = 0.2 + 0.3 * treatment_relevance
    clicked = rng.binomial(1, np.clip(true_ctr_fn, 0, 1))
    ips_result = {
        "naive_ctr": float(np.mean(clicked)),
        "ips_estimated_ctr": float(np.sum((treatment_relevance / propensity) * clicked) /
                                    np.sum(treatment_relevance / propensity)),
    }
    print(f"Toy IPS estimate of treatment policy CTR: {ips_result}")
    print("NOTE: this is illustrative only -- a real launch decision needs an actual A/B test,")
    print("      this just demonstrates the offline-estimate-before-you-ship technique.")
    RESULTS["simulated_online_ips"] = ips_result

    # =================================================================
    section("SUMMARY")
    # =================================================================
    print(json.dumps(RESULTS, indent=2, default=str))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'results_summary.json')}")
    print(f"Trained models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
