from typing import Dict, Any, List
from urva.logic.engine import LogicEngine


class HallucinationChecker:
    """
    Aggregates logic violations and conflict signals into a taxonomy of hallucinations.
    """

    def __init__(self, engine: LogicEngine, conflict_threshold: float = 0.25):
        self.engine = engine
        self.conflict_threshold = conflict_threshold

    def check_grounded(self, grounded: Dict[str, Any]) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        for fact in grounded.get("grounded_facts", []):
            text = str(fact["token"])
            v = self.engine.check_statement(text)
            violations.extend(v)
        return {"violations": violations, "has_hallucination": len(violations) > 0}

    def run_all(self, states: Dict[str, str], conflict_score: float = 0.0) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        for key in ["S1", "S2", "S3"]:
            text = states.get(key) or ""
            if not text:
                continue
            v = self.engine.apply_rules(text)
            violations.extend(v)

        has_conflict = conflict_score > self.conflict_threshold
        hallucination_type = None
        explanation = None

        if violations:
            cats = {v["category"] for v in violations}
            if "FACTUAL" in cats:
                hallucination_type = "FACTUAL"
            elif "NUMERIC" in cats:
                hallucination_type = "NUMERIC"
            elif "TEMPORAL" in cats:
                hallucination_type = "TEMPORAL"
            elif "CAUSAL" in cats:
                hallucination_type = "CAUSAL"
            elif "IMPOSSIBLE_PREMISE" in cats:
                hallucination_type = "IMPOSSIBLE PREMISE"
            else:
                hallucination_type = "LOGICAL"
            explanation = "Logic rules triggered: " + ", ".join({v["rule"] for v in violations})
        elif has_conflict:
            hallucination_type = "LOGICAL"
            explanation = "Contradictions detected between generated states."

        return {
            "violations": violations,
            "has_hallucination": bool(violations or has_conflict),
            "type": hallucination_type or "NONE",
            "explanation": explanation or "No hallucinations detected.",
        }
