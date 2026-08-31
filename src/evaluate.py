import numpy as np
from ranx import Qrels, Run, evaluate as ranx_evaluate


def evaluate_users(user_recs: dict, user_relevant: dict, k=10):
    
    qrels_dict, run_dict = {}, {}
    for uid, relevant in user_relevant.items():
        if uid not in user_recs or len(relevant) == 0:
            continue
        recs = user_recs[uid]
        qrels_dict[str(uid)] = {str(rid): 1 for rid in relevant}
        run_dict[str(uid)] = {str(rid): float(len(recs) - i) for i, rid in enumerate(recs)}

    if not qrels_dict:
        return {f"Recall@{k}": None, f"NDCG@{k}": None, f"MAP@{k}": None, "n_users_evaluated": 0}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    metrics = ranx_evaluate(qrels, run, [f"recall@{k}", f"ndcg@{k}", f"map@{k}"], make_comparable=True)

    return {
        f"Recall@{k}": float(metrics[f"recall@{k}"]),
        f"NDCG@{k}": float(metrics[f"ndcg@{k}"]),
        f"MAP@{k}": float(metrics[f"map@{k}"]),
        "n_users_evaluated": len(qrels_dict),
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
