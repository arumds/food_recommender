"""
Post-ranking constraint layer: takes the ranker's ordered list and adjusts it for
product/business needs that pure relevance-maximization ignores. This is the part
of the JD that says: "balance relevance with product and customer needs, such as
diversity, availability, business constraints and changing user intent."

Implements:
1. Availability filtering (hard constraint, applied BEFORE ranking ideally, but
   shown here as a safety-net filter too)
2. MMR (Maximal Marginal Relevance) diversification -- classic IR technique to
   avoid a top-10 that's 8 pizza places just because the user loves pizza
3. Business-rule boosting -- e.g. give small/local (non-chain) restaurants a
   visibility boost, and show the relevance/business tradeoff explicitly
"""
import numpy as np


def filter_available(candidate_df, hour):
    """Hard-filter restaurants that are closed at the given hour. Should ideally run
    at retrieval time for efficiency, but re-checking post-rank guards against staleness."""
    mask = (candidate_df["opening_hour"] <= hour) & (candidate_df["closing_hour"] > hour)
    return candidate_df[mask].reset_index(drop=True)


def mmr_rerank(candidate_df, score_col="rank_score", cuisine_col="cuisine", k=10, lambda_param=0.7):
    """
    Maximal Marginal Relevance: iteratively picks the next item that maximizes
        lambda * relevance(item) - (1 - lambda) * max_similarity(item, already_selected)
    Using cuisine match as a crude similarity proxy (same cuisine = similar = penalized).
    lambda_param close to 1 -> almost pure relevance ranking.
    lambda_param close to 0 -> maximum diversity, ignoring relevance.
    """
    remaining = candidate_df.copy().reset_index(drop=True)
    selected_idx = []
    selected_cuisines = []

    # normalize relevance score to [0, 1] for a fair tradeoff against the similarity penalty
    scores = remaining[score_col].values
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    for _ in range(min(k, len(remaining))):
        best_i, best_val = None, -np.inf
        for i in range(len(remaining)):
            if i in selected_idx:
                continue
            sim_penalty = selected_cuisines.count(remaining.at[i, cuisine_col]) / max(len(selected_cuisines), 1)
            mmr_val = lambda_param * norm_scores[i] - (1 - lambda_param) * sim_penalty
            if mmr_val > best_val:
                best_val, best_i = mmr_val, i
        selected_idx.append(best_i)
        selected_cuisines.append(remaining.at[best_i, cuisine_col])

    return remaining.iloc[selected_idx].reset_index(drop=True)


def business_boost_rerank(candidate_df, score_col="rank_score", boost_weight=0.15):
    """
    Give a visibility boost to non-chain (small/local) restaurants, simulating a
    business rule like 'support local restaurants' that trades off against pure
    relevance. Returns a NEW score column so callers can compare boosted vs raw ranking.
    """
    df = candidate_df.copy()
    scores = df[score_col].values
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    boost = np.where(df["is_chain"].values == 0, boost_weight, 0.0)
    df["boosted_score"] = norm_scores + boost
    return df.sort_values("boosted_score", ascending=False).reset_index(drop=True)


def intra_list_diversity(candidate_df, cuisine_col="cuisine"):
    """Metric: fraction of unique cuisines in the list (1.0 = all different, low = repetitive)."""
    if len(candidate_df) == 0:
        return 0.0
    return candidate_df[cuisine_col].nunique() / len(candidate_df)


def chain_share(candidate_df):
    """Metric: fraction of the list that is big chains (used to show boost's effect)."""
    if len(candidate_df) == 0:
        return 0.0
    return candidate_df["is_chain"].mean()