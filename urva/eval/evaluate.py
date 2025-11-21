from typing import Dict, Any
from tqdm import tqdm
import numpy as np


class Evaluator:
    """
    Simple evaluator: runs inference over a loader and reports aggregate scores.
    """

    def __init__(self, cfg: Dict[str, Any], pipeline):
        self.cfg = cfg
        self.pipeline = pipeline

    def run(self, loader):
        conflict_scores = []
        certainties = []
        for sample in tqdm(loader, desc="Eval"):
            out = self.pipeline.run(sample, speed="balanced")
            conflict_scores.append(out["fusion"]["conflict_score"])
            certainties.append(out["fusion"]["certainty"])
        result = {
            "mean_conflict_score": float(np.mean(conflict_scores)) if conflict_scores else 0.0,
            "mean_certainty": float(np.mean(certainties)) if certainties else 0.0,
        }
        print(result)
