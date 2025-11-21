"""
Multi-agent reasoning components.
"""
from typing import Dict, Any, List
import torch
from urva.utils.seed import set_seed
from urva.logic.engine import LogicEngine
from urva.reasoning.conflict_graph import ConflictGraph
from urva.reasoning.symbolic import SymbolicReasoner


class GeneratorAgent:
    def __init__(self, reasoner_model, seed: int = 42):
        self.model = reasoner_model
        set_seed(seed)

    def generate(self, text: str) -> Dict[str, Any]:
        return self.model({"text": text})


class LogicCriticAgent:
    def __init__(self, engine: LogicEngine):
        self.engine = engine

    def critique(self, states: Dict[str, Any]) -> List[Dict[str, Any]]:
        violations = []
        for k in ["S1", "S2", "S3"]:
            violations.extend(self.engine.apply_rules(states.get(k, "")))
        return violations


class ConsistencyVerifierAgent:
    def __init__(self, graph: ConflictGraph):
        self.graph = graph

    def verify(self, states: Dict[str, Any]) -> Dict[str, Any]:
        sentences = [s for s in [states.get("S1", ""), states.get("S2", ""), states.get("S3", "")] if s]
        return self.graph.build(sentences)


class ConfidenceAggregatorAgent:
    def aggregate(self, states: Dict[str, Any], violations: List[Dict[str, Any]], graph_score: float) -> float:
        penalty = min(1.0, 0.1 * len(violations) + graph_score)
        base = states.get("final_score", 0.5)
        return float(max(0.0, min(1.0, base * (1 - penalty))))


class RefinerAgent:
    def __init__(self, symbolic: SymbolicReasoner):
        self.symbolic = symbolic

    def refine(self, text: str, states: Dict[str, Any]) -> Dict[str, Any]:
        support = self.symbolic.forward_chain(text)
        if support["proved"]:
            states = {**states}
            states["S1"] = f"Re-evaluated: {states.get('S1','')}"
        return states
