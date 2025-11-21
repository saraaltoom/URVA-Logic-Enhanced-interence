import json
from typing import List, Dict, Any


def compute_metrics(outputs: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(len(outputs), 1)
    accurate = 0
    hallucinated = 0
    conflict_sum = 0.0
    logic_violation_sum = 0.0
    for out in outputs:
        hall = out.get("hallucination", {})
        has_hall = hall.get("has_hallucination", False)
        conflict = out.get("fusion", {}).get("conflict_score", 0.0)
        viol = hall.get("violations", [])
        if not has_hall and conflict < 0.25:
            accurate += 1
        if has_hall:
            hallucinated += 1
        conflict_sum += conflict
        logic_violation_sum += len(viol)

    return {
        "accuracy": accurate / total,
        "hallucination_rate": hallucinated / total,
        "conflict_rate": conflict_sum / total,
        "logic_violation_rate": logic_violation_sum / total,
    }


def export_json(metrics: Dict[str, float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def summarize(metrics: Dict[str, float]) -> str:
    return (
        f"Accuracy: {metrics.get('accuracy',0):.3f} | "
        f"Hallucination rate: {metrics.get('hallucination_rate',0):.3f} | "
        f"Conflict rate: {metrics.get('conflict_rate',0):.3f} | "
        f"Logic violation rate: {metrics.get('logic_violation_rate',0):.3f}"
    )
