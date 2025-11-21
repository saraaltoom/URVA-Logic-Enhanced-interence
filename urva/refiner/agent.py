from typing import Dict
from urva.states.generator import StateGenerator, StateBundle


class RefinerAgent:
    def __init__(self, seed: int = 99):
        self.generator = StateGenerator(seed=seed)

    def refine(self, refraction: Dict[str, str]) -> StateBundle:
        # regenerate with different seed to vary phrasing
        return self.generator.generate(refraction)
