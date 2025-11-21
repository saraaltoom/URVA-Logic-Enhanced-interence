from dataclasses import dataclass
from typing import List


@dataclass
class LogicRules:
    consistency_rules: List[str]
    temporal_rules: List[str]
    causal_rules: List[str]
    numeric_bounds: List[str]
    existence_rules: List[str]
    contradiction_rules: List[str]
