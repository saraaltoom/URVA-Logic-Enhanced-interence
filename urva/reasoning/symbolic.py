"""
Symbolic reasoning engine with forward/backward chaining stubs.
"""
from typing import List, Dict, Any


class SymbolicReasoner:
    def __init__(self):
        self.facts: List[str] = []
        self.rules: List[str] = []

    def add_fact(self, fact: str) -> None:
        self.facts.append(fact)

    def add_rule(self, rule: str) -> None:
        self.rules.append(rule)

    def forward_chain(self, query: str) -> Dict[str, Any]:
        derived = [f for f in self.facts if query.lower() in f.lower()]
        return {"derived": derived, "proved": bool(derived)}

    def backward_chain(self, query: str) -> Dict[str, Any]:
        proved = any(query.lower() in f.lower() for f in self.facts)
        return {"proved": proved, "support": self.facts if proved else []}
