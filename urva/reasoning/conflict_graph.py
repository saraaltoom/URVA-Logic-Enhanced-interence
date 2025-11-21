"""
Multi-Graph Contradiction Network with spectral conflict score.
"""
from typing import List, Dict, Any
import numpy as np


class ConflictGraph:
    """
    Builds pairwise relations between sentences; computes contradictions, confirmations,
    and a spectral conflict score on the adjacency matrix.
    """

    def __init__(self, lambda_conflict: float = 0.5):
        self.lambda_conflict = lambda_conflict

    def _embed(self, sent: str, dim: int = 128) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(sent)) % (2**31 - 1))
        return rng.normal(size=(dim,))

    def _rel_type(self, a: str, b: str) -> str:
        ea, eb = self._embed(a), self._embed(b)
        sim = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))
        neg_tokens = {"not", "never", "no", "cannot"}
        has_neg_a = any(tok in a.lower() for tok in neg_tokens)
        has_neg_b = any(tok in b.lower() for tok in neg_tokens)
        if sim > 0.65 and has_neg_a != has_neg_b:
            return "contradiction"
        if sim > 0.7:
            return "entailment"
        if sim < 0.2 and has_neg_a != has_neg_b:
            return "contradiction"
        return "neutral"

    def build(self, sentences: List[str]) -> Dict[str, Any]:
        edges = []
        contradictions = 0
        confirmations = 0
        n = len(sentences)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                rel = self._rel_type(sentences[i], sentences[j])
                edges.append({"a": sentences[i], "b": sentences[j], "type": rel})
                if rel == "contradiction":
                    contradictions += 1
                    adj[i, j] = adj[j, i] = -1
                elif rel == "entailment":
                    confirmations += 1
                    adj[i, j] = adj[j, i] = 1
        total = max(len(edges), 1)
        conflict_score = contradictions / total
        spectral = self._spectral_conflict(adj)
        combined = self.lambda_conflict * conflict_score + (1 - self.lambda_conflict) * spectral
        return {
            "contradictions": contradictions,
            "confirmations": confirmations,
            "total_relations": total,
            "conflict_score": combined,
            "edges": edges,
            "spectral": spectral,
        }

    def _spectral_conflict(self, adj: np.ndarray) -> float:
        if adj.size == 0:
            return 0.0
        try:
            eigs = np.linalg.eigvals(adj)
            return float(np.mean(np.abs(eigs)))
        except np.linalg.LinAlgError:
            return 0.0
