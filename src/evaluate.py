"""
Offline evaluation harness. Implements the standard IR/recsys metrics called out
in the JD: "you understand how to evaluate ML systems rigorously, including
offline metrics, experiment design and interpreting online results."

- Recall@K   : of the restaurants the user actually ordered from (in test period),
               what fraction appear in our top-K retrieved/ranked candidates?
- NDCG@K     : rewards correct items AND correct ordering (position-discounted).
- MAP@K      : mean average precision across users.

Also includes a tiny simulated "online experiment" using Inverse Propensity
Scoring (IPS) to estimate what a live A/B test might show without needing real
production traffic -- this demonstrates the offline/online metric bridge.
"""
import numpy as np


def recall_at_k(recommended_ids, relevant_ids, k):
    if len(relevant_ids) == 0:
        return None
    top_k = set(recommended_ids[:k])
    hit = len(top_k & set(relevant_ids))
    return hit / len(relevant_ids)


def ndcg_at_k(recommended_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if len(relevant_set) == 0:
        return None
    dcg = 0.0
    for i, rid in enumerate(recommended_ids[:k]):
        if rid in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # rank starts at 1, so i+2 for log base
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    if len(relevant_set) == 0:
        return None
    hits, sum_prec = 0, 0.0
    for i, rid in enumerate(recommended_ids[:k]):
        if rid in relevant_set:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant_set), k)


def evaluate_users(user_recs: dict, user_relevant: dict, k=10):
    """
    user_recs:     {user_id: [ranked restaurant_ids]}
    user_relevant: {user_id: [restaurant_ids the user actually ordered from in test period]}
    Returns mean Recall@K, NDCG@K, MAP@K across users that had at least one relevant item.
    """
    recalls, ndcgs, maps = [], [], []
    for uid, relevant in user_relevant.items():
        if uid not in user_recs or len(relevant) == 0:
            continue
        recs = user_recs[uid]
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        maps.append(average_precision_at_k(recs, relevant, k))
    return {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else None,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else None,
        f"MAP@{k}": float(np.mean(maps)) if maps else None,
        "n_users_evaluated": len(recalls),
    }


def simulate_online_lift_ips(logged_scores_control, logged_scores_treatment, clicked, propensity):
    """
    Toy Inverse-Propensity-Scoring estimator: given logged (control-policy) data where
    we know the probability the control policy would have shown each item (propensity),
    estimate what click-through/order rate the treatment (new model) policy would achieve.
    This is a simplified illustration of the offline->online bridge, NOT a substitute
    for a real A/B test -- the writeup should flag that explicitly.
    """
    weights = logged_scores_treatment / np.clip(propensity, 1e-3, None)
    ips_estimate = np.sum(weights * clicked) / np.sum(weights)
    naive_estimate = np.mean(clicked)
    return {"naive_ctr": float(naive_estimate), "ips_estimated_ctr": float(ips_estimate)}