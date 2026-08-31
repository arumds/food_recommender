import numpy as np
import pandas as pd
import lightgbm as lgb

CUISINES = [
    "pizza", "sushi", "burgers", "vegan", "indian",
    "thai", "italian", "chinese", "kebab", "salads",
]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_ranking_features(interactions, users, restaurants, embedding_retriever=None):
    """
    Join interaction events with user/restaurant features to build a training table.
    Each row = one (user, restaurant) candidate pair with a relevance label (ordered 0/1).
    """
    df = interactions.merge(users, on="user_id", suffixes=("", "_u"))
    df = df.merge(restaurants, on="restaurant_id", suffixes=("", "_r"))

    # personalization feature: does this user's affinity vector favor this restaurant's cuisine?
    def affinity_lookup(row):
        return row[f"affinity_{row['cuisine']}"]
    df["user_cuisine_affinity"] = df.apply(affinity_lookup, axis=1)

    df["distance_km"] = _haversine_km(df["lat"], df["lon"], df["lat_r"], df["lon_r"])

    # lunch/dinner interaction: some cuisines spike at specific hours
    df["is_lunch_hour"] = ((df["hour"] >= 11) & (df["hour"] <= 14)).astype(int)
    df["is_dinner_hour"] = ((df["hour"] >= 18) & (df["hour"] <= 21)).astype(int)

    # retrieval/embedding score as a ranking feature (if provided)
    if embedding_retriever is not None:
        u_emb = embedding_retriever.user_emb[df["user_id"].values]
        r_emb = embedding_retriever.rest_emb[df["restaurant_id"].values]
        df["embedding_score"] = np.sum(u_emb * r_emb, axis=1)
    else:
        df["embedding_score"] = 0.0

    feature_cols = [
        "user_cuisine_affinity", "distance_km", "popularity", "rating",
        "price_tier", "avg_delivery_min", "is_chain", "is_lunch_hour",
        "is_dinner_hour", "embedding_score",
    ]
    return df, feature_cols


def time_split(df, test_frac=0.2):
    """Time-based split (not random!) -- train on older interactions, test on the most
    recent ones. This avoids leakage and mirrors how you'd actually evaluate in production,
    where the model is always predicting the future from the past."""
    cutoff = df["timestamp_rank"].quantile(1 - test_frac)
    train = df[df["timestamp_rank"] < cutoff].copy()
    test = df[df["timestamp_rank"] >= cutoff].copy()
    return train, test


def train_ranker(train_df, feature_cols, label_col="ordered", group_col="user_id"):
    """Train a LightGBM LambdaMART ranker. Requires rows to be sorted / grouped by query (user)."""
    train_df = train_df.sort_values(group_col)
    group_sizes = train_df.groupby(group_col).size().values

    train_set = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[label_col],
        group=group_sizes,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 50,  # heavier regularization -- small per-user groups overfit fast
        "verbosity": -1,
    }
    model = lgb.train(params, train_set, num_boost_round=40)
    return model


def build_candidate_features(user_ids, candidate_lists, users, restaurants, hour, embedding_retriever=None):
    """
    Build a scoring table for arbitrary (user, candidate_restaurant) pairs -- i.e. the
    output of a RETRIEVAL stage, not just restaurants the user happened to interact
    with historically. This is what a real ranker scores at serve time: a candidate
    pool per user, most of which the user has never ordered from before.

    user_ids:       list of user_id
    candidate_lists: list of lists, candidate_lists[i] = restaurant_ids retrieved for user_ids[i]
    """
    rows_user, rows_rest = [], []
    for uid, cands in zip(user_ids, candidate_lists):
        rows_user.extend([uid] * len(cands))
        rows_rest.extend(cands)

    df = pd.DataFrame({"user_id": rows_user, "restaurant_id": rows_rest})
    df = df.merge(users, on="user_id", suffixes=("", "_u"))
    df = df.merge(restaurants, on="restaurant_id", suffixes=("", "_r"))

    def affinity_lookup(row):
        return row[f"affinity_{row['cuisine']}"]
    df["user_cuisine_affinity"] = df.apply(affinity_lookup, axis=1)
    df["distance_km"] = _haversine_km(df["lat"], df["lon"], df["lat_r"], df["lon_r"])
    df["is_lunch_hour"] = int(11 <= hour <= 14)
    df["is_dinner_hour"] = int(18 <= hour <= 21)

    if embedding_retriever is not None:
        u_emb = embedding_retriever.user_emb[df["user_id"].values]
        r_emb = embedding_retriever.rest_emb[df["restaurant_id"].values]
        df["embedding_score"] = np.sum(u_emb * r_emb, axis=1)
    else:
        df["embedding_score"] = 0.0

    return df


def train_baseline_popularity_ranker():
    """A trivial 'ranker' that just returns popularity as the score -- used as a
    non-personalized baseline to prove the learned ranker adds value."""
    def score_fn(df, feature_cols):
        return df["popularity"].values
    return score_fn