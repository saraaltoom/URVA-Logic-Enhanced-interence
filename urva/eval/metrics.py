import json
from typing import List, Dict, Any
import numpy as np


def _compute_certainty(logic_penalty: float, faithfulness: float,
                       grounding: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Certainty = (1-L)(αF + βG) — Equation 1 & 3 from paper"""
    return max(0.0, (1.0 - logic_penalty) * (alpha * faithfulness + beta * grounding))


def compute_metrics(outputs: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(len(outputs), 1)
    accurate = 0
    hallucinated = 0
    conflict_sum = 0.0
    logic_violation_sum = 0.0
    spectral_sum = 0.0
    certainties = []
    cons_scores = []

    for out in outputs:
        hall = out.get("hallucination", {})
        has_hall = hall.get("has_hallucination", False)
        conflict = out.get("fusion", {}).get("conflict_score", 0.0)
        spectral = out.get("conflict_graph", {}).get("spectral", 0.0)
        viol = hall.get("violations", [])

        # Compute certainty per example using paper formula
        pred = out.get("final_answer", "")
        ctx  = out.get("fusion", {}).get("context_match", "")
        pred_tok = set(pred.lower().split())
        ctx_tok  = set(ctx.lower().split())
        _F = len(pred_tok & ctx_tok) / max(len(ctx_tok), 1)
        _G = out.get("grounding", {}).get("avg_score", 0.0)
        _L = min(len(viol) / 5, 1.0)
        certainty = _compute_certainty(_L, _F, _G)

        if not has_hall and conflict < 0.25:
            accurate += 1
        if has_hall:
            hallucinated += 1
        conflict_sum += conflict
        spectral_sum += spectral
        logic_violation_sum += len(viol)
        certainties.append(certainty)
        cons_scores.append(out.get("fusion", {}).get("reasoning_alignment", 0.0))

    consistency_var = float(np.var(cons_scores)) if cons_scores else 0.0
    calibration_error = (
        float(np.abs(np.array(certainties) -
                     np.array([1 - conflict_sum / total] * len(certainties))).mean())
        if certainties else 0.0
    )

    return {
        "accuracy": accurate / total,
        "hallucination_rate": hallucinated / total,
        "conflict_rate": conflict_sum / total,
        "logic_violation_rate": logic_violation_sum / total,
        "spectral_conflict": spectral_sum / total,
        "self_consistency_variance": consistency_var,
        "calibration_error": calibration_error,
        "avg_certainty": float(np.mean(certainties))
    }


def export_json(metrics: Dict[str, float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def summarize(metrics: Dict[str, float]) -> str:
    return (
        f"Acc: {metrics.get('accuracy', 0):.3f} | "
        f"Halluc: {metrics.get('hallucination_rate', 0):.3f} | "
        f"Conflict: {metrics.get('conflict_rate', 0):.3f} | "
        f"Spectral: {metrics.get('spectral_conflict', 0):.3f} | "
        f"Logic Viol: {metrics.get('logic_violation_rate', 0):.3f} | "
        f"Cons Var: {metrics.get('self_consistency_variance', 0):.3f} | "
        f"Calib Err: {metrics.get('calibration_error', 0):.3f}"
    )
