"""
Optimizer factory for URVA.
"""
from typing import Dict, Any
import torch
from torch import optim


def build_optimizer(params, cfg: Dict[str, Any]) -> optim.Optimizer:
    name = cfg.get("optimizer", {}).get("name", "adamw").lower()
    lr = cfg.get("learning_rate", 1e-3)
    wd = cfg.get("weight_decay", 0.01)
    betas = tuple(cfg.get("optimizer", {}).get("betas", (0.9, 0.999)))
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")
