"""
Refiner agent that can rephrase or adjust states; for compatibility with legacy calls.
In the new pipeline, refinement is handled in urva.reasoning.agents.RefinerAgent.
"""
from typing import Dict
from urva.states.generator import StateGenerator, StateBundle


class RefinerAgent:
    def __init__(self, seed: int = 99):
        self.generator = StateGenerator(seed=seed)

    def refine(self, refraction: Dict[str, str]) -> StateBundle:
        return self.generator.generate(refraction)
