"""
Scheduler factory for URVA.
"""
from typing import Dict, Any
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    name = cfg.get("scheduler", {}).get("name", "cosine")
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(cfg.get("num_epochs", 1), 1))
    if name == "warmup":
        warmup_steps = cfg.get("scheduler", {}).get("warmup_steps", 100)

        def lr_lambda(step):
            return min(1.0, (step + 1) / float(warmup_steps))

        return LambdaLR(optimizer, lr_lambda)
    return None
