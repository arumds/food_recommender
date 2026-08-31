import time
import numpy as np
import faiss


class FlatIPRetriever:
    """Exact inner-product (cosine, since embeddings are L2-normalized) search."""

    def __init__(self, item_embeddings: np.ndarray):
        self.dim = item_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(item_embeddings.astype(np.float32))

    def retrieve(self, user_embedding: np.ndarray, k=50):
        scores, idx = self.index.search(user_embedding.reshape(1, -1).astype(np.float32), k)
        return idx[0].tolist()


class HNSWRetriever:
    """Approximate NN search via HNSW graph -- what you'd actually deploy at scale."""

    def __init__(self, item_embeddings: np.ndarray, M=32, ef_construction=100, ef_search=64):
        self.dim = item_embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.index.add(item_embeddings.astype(np.float32))

    def retrieve(self, user_embedding: np.ndarray, k=50):
        scores, idx = self.index.search(user_embedding.reshape(1, -1).astype(np.float32), k)
        return idx[0].tolist()


def benchmark_retrievers(user_embeddings: np.ndarray, item_embeddings: np.ndarray, k=50, n_queries=300):
    """
    Compares brute-force numpy, FAISS FlatIP (exact), and FAISS HNSW (approximate)
    on: (a) recall of HNSW vs the exact FlatIP result (i.e. how much accuracy is
    traded away), and (b) query latency, at this demo's ~500-item scale.
    """
    flat = FlatIPRetriever(item_embeddings)
    hnsw = HNSWRetriever(item_embeddings)

    n_queries = min(n_queries, user_embeddings.shape[0])
    query_ids = np.arange(n_queries)

    # latency: brute-force numpy
    t0 = time.perf_counter()
    for uid in query_ids:
        scores = item_embeddings @ user_embeddings[uid]
        _ = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    t_numpy = time.perf_counter() - t0

    # latency + results: FAISS exact
    t0 = time.perf_counter()
    flat_results = [flat.retrieve(user_embeddings[uid], k=k) for uid in query_ids]
    t_flat = time.perf_counter() - t0

    # latency + results: FAISS HNSW (approximate)
    t0 = time.perf_counter()
    hnsw_results = [hnsw.retrieve(user_embeddings[uid], k=k) for uid in query_ids]
    t_hnsw = time.perf_counter() - t0

    # recall of HNSW against the exact FlatIP result (treated as ground truth for this check)
    overlaps = [
        len(set(hnsw_results[i]) & set(flat_results[i])) / k
        for i in range(n_queries)
    ]

    return {
        "n_queries": n_queries,
        "numpy_brute_force_sec": t_numpy,
        "faiss_flat_exact_sec": t_flat,
        "faiss_hnsw_approx_sec": t_hnsw,
        "hnsw_recall_vs_exact": float(np.mean(overlaps)),
    }