"""
Simple retrieval and vector store stubs for retrieval-augmented reasoning.
"""
from typing import List, Tuple
import numpy as np


class VectorStore:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.docs: List[Tuple[str, np.ndarray]] = []

    def add(self, doc_id: str, text: str) -> None:
        emb = self.embed(text)
        self.docs.append((doc_id, emb))

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31 - 1))
        return rng.normal(size=(self.dim,))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_emb = self.embed(query)
        sims = []
        for doc_id, emb in self.docs:
            sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
            sims.append((doc_id, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]


def retrieve_topk(store: VectorStore, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    return store.search(query, top_k=top_k)
