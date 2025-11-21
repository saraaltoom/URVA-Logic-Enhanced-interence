"""
Benchmark evaluator producing aggregate metrics and optional plotting hooks.
"""
from typing import Dict, Any
from tqdm import tqdm
import numpy as np

from urva.eval.metrics import compute_metrics, summarize


class Evaluator:
    def __init__(self, cfg: Dict[str, Any], pipeline):
        self.cfg = cfg
        self.pipeline = pipeline

    def run(self, loader, speed: str = "balanced"):
        outputs = []
        for sample in tqdm(loader, desc="Eval"):
            out = self.pipeline.run(sample, speed=speed)
            outputs.append(out)
        metrics = compute_metrics(outputs)
        print(summarize(metrics))
        return metrics
