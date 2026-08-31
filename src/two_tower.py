import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CUISINES = [
    "pizza", "sushi", "burgers", "vegan", "indian",
    "thai", "italian", "chinese", "kebab", "salads",
]
N_CUISINES = len(CUISINES)


class Tower(nn.Module):
    """Simple MLP tower: raw features -> shared embedding space."""

    def __init__(self, input_dim, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=-1)  # L2-normalize -> dot product = cosine sim


class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, embedding_dim=32):
        super().__init__()
        self.user_tower = Tower(user_input_dim, embedding_dim)
        self.item_tower = Tower(item_input_dim, embedding_dim)

    def forward(self, user_x, item_x):
        u_emb = self.user_tower(user_x)
        i_emb = self.item_tower(item_x)
        return (u_emb * i_emb).sum(dim=-1)  # cosine similarity score

    def embed_users(self, user_x):
        return self.user_tower(user_x)

    def embed_items(self, item_x):
        return self.item_tower(item_x)


def build_user_features(users_df):
    """[cuisine affinities (10), normalized signup recency, lat, lon] -> (n_users, 13)"""
    affinity_cols = [f"affinity_{c}" for c in CUISINES]
    aff = users_df[affinity_cols].values.astype(np.float32)
    recency = (users_df["signup_days_ago"].values / 400.0).astype(np.float32).reshape(-1, 1)
    lat = ((users_df["lat"].values - 60.2) * 10).astype(np.float32).reshape(-1, 1)  # rescale for NN stability
    lon = ((users_df["lon"].values - 24.95) * 10).astype(np.float32).reshape(-1, 1)
    return np.hstack([aff, recency, lat, lon])


def build_item_features(restaurants_df):
    """[one-hot cuisine (10), popularity, rating/5, price_tier/3, delivery/60, is_chain, lat, lon] -> (n_items, 17)"""
    cuisine_idx = {c: i for i, c in enumerate(CUISINES)}
    n = len(restaurants_df)
    onehot = np.zeros((n, N_CUISINES), dtype=np.float32)
    for i, c in enumerate(restaurants_df["cuisine"].values):
        onehot[i, cuisine_idx[c]] = 1.0
    popularity = restaurants_df["popularity"].values.astype(np.float32).reshape(-1, 1)
    rating = (restaurants_df["rating"].values / 5.0).astype(np.float32).reshape(-1, 1)
    price = (restaurants_df["price_tier"].values / 3.0).astype(np.float32).reshape(-1, 1)
    delivery = (restaurants_df["avg_delivery_min"].values / 60.0).astype(np.float32).reshape(-1, 1)
    is_chain = restaurants_df["is_chain"].values.astype(np.float32).reshape(-1, 1)
    lat = ((restaurants_df["lat"].values - 60.2) * 10).astype(np.float32).reshape(-1, 1)
    lon = ((restaurants_df["lon"].values - 24.95) * 10).astype(np.float32).reshape(-1, 1)
    return np.hstack([onehot, popularity, rating, price, delivery, is_chain, lat, lon])


def train_two_tower(
    interactions_df, users_df, restaurants_df,
    embedding_dim=32, epochs=8, batch_size=512, lr=1e-3, neg_ratio=4, device="cpu",
):
    user_feat = torch.tensor(build_user_features(users_df), dtype=torch.float32, device=device)
    item_feat = torch.tensor(build_item_features(restaurants_df), dtype=torch.float32, device=device)

    model = TwoTowerModel(user_feat.shape[1], item_feat.shape[1], embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_pairs = interactions_df[interactions_df["ordered"] == 1][["user_id", "restaurant_id"]].values
    n_items = item_feat.shape[0]
    rng = np.random.default_rng(0)

    for epoch in range(epochs):
        perm = rng.permutation(len(pos_pairs))
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            batch_pos = pos_pairs[batch_idx]
            u_ids = batch_pos[:, 0]
            pos_i_ids = batch_pos[:, 1]

            # build a batch of positives + sampled negatives per positive
            neg_i_ids = rng.integers(0, n_items, size=(len(u_ids), neg_ratio))
            all_u_ids = np.repeat(u_ids, 1 + neg_ratio)
            all_i_ids = np.concatenate([pos_i_ids.reshape(-1, 1), neg_i_ids], axis=1).reshape(-1)
            labels = np.zeros(len(all_u_ids), dtype=np.float32)
            labels[0::(1 + neg_ratio)] = 1.0  # first of every group is the positive

            u_x = user_feat[torch.tensor(all_u_ids, dtype=torch.long)]
            i_x = item_feat[torch.tensor(all_i_ids, dtype=torch.long)]
            y = torch.tensor(labels, device=device)

            optimizer.zero_grad()
            scores = model(u_x, i_x)
            loss = F.binary_cross_entropy_with_logits(scores * 5.0, y)  # temperature scale
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"  [TwoTower] epoch {epoch+1}/{epochs}  loss={total_loss/n_batches:.4f}")

    return model, user_feat, item_feat


class TwoTowerRetriever:
    """Wraps a trained TwoTowerModel + precomputed item embeddings for retrieval."""

    def __init__(self, model, user_feat, item_feat):
        self.model = model
        self.model.eval()
        with torch.no_grad():
            self.user_emb = model.embed_users(user_feat).numpy()
            self.item_emb = model.embed_items(item_feat).numpy()

    def retrieve(self, user_id, k=50):
        scores = self.item_emb @ self.user_emb[user_id]
        top_k = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return top_k[np.argsort(-scores[top_k])].tolist()