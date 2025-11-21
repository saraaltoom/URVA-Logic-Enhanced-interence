from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ConflictResult:
    contradictions: int
    confirmations: int
    total_relations: int
    conflict_score: float
    edges: List[Dict[str, str]]


class ContradictionGraph:
    def __init__(self):
        pass

    @staticmethod
    def _sentences(text: str) -> List[str]:
        parts = [p.strip() for p in text.replace("!", ".").replace("?", ".").split(".")]
        return [p for p in parts if p]

    def build(self, states: List[str]) -> ConflictResult:
        nodes = []
        for idx, st in enumerate(states):
            for sent in self._sentences(st):
                nodes.append((idx, sent.lower()))

        contradictions = 0
        confirmations = 0
        edges: List[Dict[str, str]] = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a_idx, a = nodes[i]
                b_idx, b = nodes[j]
                if not a or not b:
                    continue
                if a == b:
                    confirmations += 1
                    edges.append({"type": "confirm", "a": a, "b": b})
                elif self._is_contradiction(a, b):
                    contradictions += 1
                    edges.append({"type": "contradict", "a": a, "b": b})

        total_relations = contradictions + confirmations
        conflict_score = contradictions / total_relations if total_relations else 0.0
        return ConflictResult(
            contradictions=contradictions,
            confirmations=confirmations,
            total_relations=total_relations,
            conflict_score=conflict_score,
            edges=edges,
        )

    def _is_contradiction(self, a: str, b: str) -> bool:
        # naive contradiction: presence of "not" in one but not the other, referring to same phrase
        tokens = set(a.split())
        common = tokens.intersection(set(b.split()))
        if len(common) < 2:
            return False
        has_neg_a = "not" in a or "no" in a
        has_neg_b = "not" in b or "no" in b
        return has_neg_a != has_neg_b
