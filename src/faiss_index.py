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
