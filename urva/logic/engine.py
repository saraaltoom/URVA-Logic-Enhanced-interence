import json
from typing import Dict, List, Any
from .rules import LogicRules


class LogicEngine:
    """
    Expanded logic engine that checks multiple rule families and returns structured violations.
    """

    def __init__(self, rules: LogicRules):
        self.rules = rules

    @classmethod
    def from_file(cls, path: str) -> "LogicEngine":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rules = LogicRules(**data)
        return cls(rules)

    def apply_rules(self, text: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        low = text.lower()

        def add(cat: str, rule: str, severity: str, message: str):
            results.append(
                {"category": cat, "rule": rule, "severity": severity, "message": message}
            )

        # Factual consistency: detect double negations or self-contradiction heuristics
        if "not" in low and "never" in low:
            for r in self.rules.consistency_rules:
                add("FACTUAL", r, "medium", "Conflicting negations detected.")

        # Numeric bounds
        if any(tok in low for tok in ["-km", " impossible distance", "lightyears in an atom"]):
            for r in self.rules.numeric_bounds:
                add("NUMERIC", r, "high", "Numeric bounds exceeded or invalid.")

        # Temporal rules
        if any(tok in low for tok in ["in the future", "2100", "next century", "before birth"]):
            for r in self.rules.temporal_rules:
                add("TEMPORAL", r, "medium", "Temporal feasibility violated.")

        # Entity existence
        if any(tok in low for tok in ["unicorn", "hogwarts", "atlantis", "mythical"]):
            for r in self.rules.existence_rules:
                add("EXISTENCE", r, "medium", "Nonexistent or fictional entity detected.")

        # Causal feasibility
        if "without cause" in low or "effect before cause" in low:
            for r in self.rules.causal_rules:
                add("CAUSAL", r, "high", "Cause/effect ordering violated.")

        # Impossible premise
        if any(tok in low for tok in ["perpetual motion", "faster than light", "square circle"]):
            add("IMPOSSIBLE_PREMISE", "Physical impossibility", "critical", "Impossible premise detected.")

        return results

    def check_statement(self, statement: str) -> List[Dict[str, Any]]:
        return self.apply_rules(statement)

    def summarize(self) -> str:
        return json.dumps(self.rules.__dict__, indent=2)
