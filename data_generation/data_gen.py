import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CUISINES = [
    "pizza", "sushi", "burgers", "vegan", "indian",
    "thai", "italian", "chinese", "kebab", "salads",
]
N_CUISINES = len(CUISINES)


def generate_users(n_users: int = 2000) -> pd.DataFrame:
    """Each user has a latent affinity vector over cuisines (their 'taste embedding')."""
    # Dirichlet gives each user a peaky-ish preference distribution over cuisines,
    # similar to how real users tend to favor 2-3 cuisines rather than being uniform.
    affinities = RNG.dirichlet(alpha=np.full(N_CUISINES, 0.4), size=n_users)
    lat = RNG.uniform(60.15, 60.25, n_users)   # Helsinki-ish bounding box
    lon = RNG.uniform(24.85, 25.05, n_users)
    signup_days_ago = RNG.integers(0, 400, n_users)
    users = pd.DataFrame({
        "user_id": np.arange(n_users),
        "lat": lat,
        "lon": lon,
        "signup_days_ago": signup_days_ago,
    })
    for i, c in enumerate(CUISINES):
        users[f"affinity_{c}"] = affinities[:, i]
    return users


def generate_restaurants(n_restaurants: int = 500) -> pd.DataFrame:
    cuisine = RNG.choice(CUISINES, size=n_restaurants)
    # Popularity follows a power law -- a few chains dominate, long tail of small places.
    popularity = RNG.pareto(a=1.5, size=n_restaurants) + 0.1
    popularity = popularity / popularity.max()
    price_tier = RNG.integers(1, 4, n_restaurants)  # 1=cheap, 3=expensive
    rating = np.clip(RNG.normal(4.2, 0.4, n_restaurants), 2.5, 5.0)
    avg_delivery_min = RNG.integers(15, 55, n_restaurants)
    lat = RNG.uniform(60.15, 60.25, n_restaurants)
    lon = RNG.uniform(24.85, 25.05, n_restaurants)
    # is_chain flags a "big business" restaurant (used later for the business-constraint demo)
    is_chain = (popularity > np.quantile(popularity, 0.85)).astype(int)
    # opening_hour/closing_hour lets us simulate "closed right now" filtering
    opening_hour = RNG.integers(7, 12, n_restaurants)
    closing_hour = RNG.integers(20, 24, n_restaurants)

    return pd.DataFrame({
        "restaurant_id": np.arange(n_restaurants),
        "cuisine": cuisine,
        "popularity": popularity,
        "price_tier": price_tier,
        "rating": rating,
        "avg_delivery_min": avg_delivery_min,
        "lat": lat,
        "lon": lon,
        "is_chain": is_chain,
        "opening_hour": opening_hour,
        "closing_hour": closing_hour,
    })


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def generate_interactions(
    users: pd.DataFrame,
    restaurants: pd.DataFrame,
    n_interactions: int = 60_000,
) -> pd.DataFrame:
    """
    Simulate implicit-feedback events. For each event we sample a user, then bias the
    restaurant choice by: (a) the user's cuisine affinity, (b) restaurant popularity,
    (c) distance. This creates real learnable structure for retrieval/ranking models.
    """
    n_users, n_rest = len(users), len(restaurants)
    cuisine_idx = {c: i for i, c in enumerate(CUISINES)}
    rest_cuisine_idx = restaurants["cuisine"].map(cuisine_idx).values

    rows = []
    user_ids = RNG.integers(0, n_users, n_interactions)
    hours = RNG.integers(8, 23, n_interactions)
    days_ago = RNG.integers(0, 90, n_interactions)  # 90-day interaction window

    affinity_cols = [f"affinity_{c}" for c in CUISINES]
    user_affinity_matrix = users[affinity_cols].values  # (n_users, n_cuisines)

    for k in range(n_interactions):
        uid = user_ids[k]
        u_lat, u_lon = users.at[uid, "lat"], users.at[uid, "lon"]

        # EXPOSURE model: which restaurant does the user even see/consider? Mixes affinity,
        # popularity and distance with a fair amount of exploration noise -- like a logging
        # policy that doesn't perfectly pre-personalize, so the outcome model below has
        # non-degenerate variation to learn from (avoids the "everything shown is already
        # a good match" selection-bias trap that would make ranking trivial or unlearnable).
        aff_score = user_affinity_matrix[uid][rest_cuisine_idx]
        dist_km = _haversine_km(u_lat, u_lon, restaurants["lat"].values, restaurants["lon"].values)
        dist_penalty = np.exp(-dist_km / 4.0)
        exposure_score = (
            1.0 * aff_score
            + 1.0 * restaurants["popularity"].values
            + 1.2 * dist_penalty
            + RNG.normal(0, 0.3, n_rest)
        )
        probs = np.exp(exposure_score * 2) / np.exp(exposure_score * 2).sum()
        rid = RNG.choice(n_rest, p=probs)

        # OUTCOME model: given exposure, did it convert to an order? Deliberately dominated
        # by cuisine-affinity match (the personalization signal) with modest contributions
        # from popularity, and a base rate floor -- this is the ground truth the ranker
        # has to recover from features.
        base_p_order = 0.08 + 0.75 * aff_score[rid] + 0.07 * restaurants.at[rid, "popularity"]
        ordered = int(RNG.random() < min(base_p_order, 0.95))

        rows.append((uid, rid, days_ago[k], hours[k], ordered))

    df = pd.DataFrame(rows, columns=["user_id", "restaurant_id", "days_ago", "hour", "ordered"])
    df["timestamp_rank"] = -df["days_ago"]  # higher = more recent, for time split
    return df


def build_dataset(n_users=2000, n_restaurants=500, n_interactions=60_000):
    users = generate_users(n_users)
    restaurants = generate_restaurants(n_restaurants)
    interactions = generate_interactions(users, restaurants, n_interactions)
    return users, restaurants, interactions


if __name__ == "__main__":
    import os

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    users, restaurants, interactions = build_dataset(n_users=1500, n_restaurants=400, n_interactions=45_000)

    users.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
    restaurants.to_csv(os.path.join(DATA_DIR, "restaurants.csv"), index=False)
    interactions.to_csv(os.path.join(DATA_DIR, "interactions.csv"), index=False)

    print(f"Data written to: {DATA_DIR}")
    print(f"users: {users.shape}, restaurants: {restaurants.shape}, interactions: {interactions.shape}")
    print(f"order rate: {interactions['ordered'].mean():.3f}")
    print("\nData generation complete. Run the pipeline with:\n  python3 src/main.py")