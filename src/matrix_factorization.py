"""
Matrix factorization baseline (implicit ALS, Hu/Koren/Volinsky 2008).

Pure collaborative filtering: only uses the user x restaurant order matrix,
no user or restaurant features. Users with no orders get popular restaurants.
"""
import numpy as np
import scipy.sparse as sp


def build_interaction_matrix(interactions_df, n_users, n_items):
    orders = interactions_df[interactions_df["ordered"] == 1]
    counts = orders.groupby(["user_id", "restaurant_id"]).size()
    rows = counts.index.get_level_values(0).values
    cols = counts.index.get_level_values(1).values
    return sp.csr_matrix(
        (counts.values.astype(np.float32), (rows, cols)), shape=(n_users, n_items)
    )


def _als_step(R, Y, alpha, reg):
    # solve each row's factors with Y fixed; YtY is shared across rows
    n_rows, k = R.shape[0], Y.shape[1]
    YtY = Y.T @ Y
    reg_eye = reg * np.eye(k, dtype=np.float64)
    X = np.zeros((n_rows, k), dtype=np.float64)
    for u in range(n_rows):
        start, end = R.indptr[u], R.indptr[u + 1]
        if start == end:
            continue
        idx = R.indices[start:end]
        conf = 1.0 + alpha * R.data[start:end]
        Y_u = Y[idx]
        A = YtY + (Y_u.T * (conf - 1.0)) @ Y_u + reg_eye
        b = Y_u.T @ conf
        X[u] = np.linalg.solve(A, b)
    return X


class ImplicitALS:
    def __init__(self, n_factors=32, reg=0.1, alpha=20.0, iterations=15, seed=0):
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors = None
        self.item_factors = None

    def fit(self, R: sp.csr_matrix, verbose=True):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = R.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))
        Rt = R.T.tocsr()
        for it in range(self.iterations):
            self.user_factors = _als_step(R, self.item_factors, self.alpha, self.reg)
            self.item_factors = _als_step(Rt, self.user_factors, self.alpha, self.reg)
            if verbose and (it == 0 or (it + 1) % 5 == 0):
                print(f"  [ALS] iteration {it+1}/{self.iterations}  "
                      f"weighted loss={self._loss(R):.1f}")
        self.user_factors = self.user_factors.astype(np.float32)
        self.item_factors = self.item_factors.astype(np.float32)
        return self

    def _loss(self, R):
        pred = self.user_factors @ self.item_factors.T
        P = (R.toarray() > 0).astype(np.float64)
        C = 1.0 + self.alpha * R.toarray()
        reg = self.reg * ((self.user_factors ** 2).sum() + (self.item_factors ** 2).sum())
        return float((C * (P - pred) ** 2).sum() + reg)

    def recommend(self, user_id, k=20):
        scores = self.item_factors @ self.user_factors[user_id]
        top_k = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return top_k[np.argsort(-scores[top_k])].tolist()


class MFRetriever:
    def __init__(self, als: ImplicitALS, R: sp.csr_matrix):
        self.als = als
        self.user_n_orders = np.asarray(R.sum(axis=1)).ravel()
        self.popular = np.argsort(-np.asarray(R.sum(axis=0)).ravel()).tolist()

    def is_cold(self, user_id):
        return self.user_n_orders[user_id] == 0

    def retrieve(self, user_id, k=20):
        if self.is_cold(user_id):
            return self.popular[:k]
        return self.als.recommend(user_id, k)


def popularity_recs(R: sp.csr_matrix, user_ids, k=20):
    top = np.argsort(-np.asarray(R.sum(axis=0)).ravel())[:k].tolist()
    return {uid: top for uid in user_ids}
