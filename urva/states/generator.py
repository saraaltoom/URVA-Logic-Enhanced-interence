"""
State generator for backward compatibility with legacy refraction/refiner paths.
"""
from dataclasses import dataclass
from typing import Dict
import hashlib
import random


@dataclass
class StateBundle:
    S1: str
    S2: str
    S3: str


class StateGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def _rng(self, key: str) -> random.Random:
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
        return random.Random(h + self.seed)

    def generate(self, refraction: Dict[str, str]) -> StateBundle:
        base = refraction.get("Q_raw", "")
        rng = self._rng(base)
        templates = [
            "Direct: {t}",
            "Answer: {t}",
            "Result: {t}",
        ]
        reasons = [
            "Reasoning: derived from context and logic on '{t}'.",
            "Explained: considering constraints in '{t}'.",
            "Rationale: traced implications in '{t}'.",
        ]
        justs = [
            "Justification: aligned with known facts around '{t}'.",
            "Support: ties back to rules and evidence in '{t}'.",
            "Grounding: checked against consistency with '{t}'.",
        ]
        s1 = templates[rng.randrange(len(templates))].format(t=base)
        s2 = reasons[rng.randrange(len(reasons))].format(t=base)
        s3 = justs[rng.randrange(len(justs))].format(t=base)
        return StateBundle(S1=s1, S2=s2, S3=s3)
