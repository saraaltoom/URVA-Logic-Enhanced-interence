import json
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cpu",
    "batch_size": 8,
    "num_epochs": 1,
    "learning_rate": 1e-3,
    "hidden_size": 128,
    "dropout": 0.1,
    "max_hops": 3,
    "grounder": {"threshold": 0.5},
    "reasoner": {"max_depth": 3},
    "checker": {"max_violations": 3},
    "graph": {"conflict_threshold": 0.25},
    "refine_loops": {"aggressive": 2, "smart": 1, "turbo": 0},
}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(user_cfg)
    return cfg
