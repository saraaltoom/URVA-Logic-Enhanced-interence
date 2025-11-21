from typing import List, Dict, Any
from urva.eval.metrics import compute_metrics
from urva.baselines.gpt_baseline import run_gpt_baseline


def _evaluate_baseline(dataset: List[Dict[str, Any]], logic) -> List[Dict[str, Any]]:
    outputs = []
    for sample in dataset:
        ans = run_gpt_baseline(sample.get("text", ""))
        violations = logic.apply_rules(ans["answer"]) if logic else []
        has_hall = bool(violations)
        outputs.append(
            {
                "id": sample.get("id"),
                "final_answer": ans["answer"],
                "hallucination": {"violations": violations, "has_hallucination": has_hall},
                "fusion": {"conflict_score": 0.0},
            }
        )
    return outputs


def compare_urva_vs_gpt(dataset: List[Dict[str, Any]], urva_pipeline, logic, cfg, speed: str = "balanced", ablation=None):
    urva_outputs = []
    for sample in dataset:
        urva_outputs.append(urva_pipeline.run(sample, speed=speed, ablation=ablation))

    gpt_outputs = _evaluate_baseline(dataset, logic)

    urva_metrics = compute_metrics(urva_outputs)
    gpt_metrics = compute_metrics(gpt_outputs)

    def safe_diff(a, b):
        return a - b

    halluc_improvement = gpt_metrics["hallucination_rate"] - urva_metrics["hallucination_rate"]
    conflict_diff = gpt_metrics["conflict_rate"] - urva_metrics["conflict_rate"]
    accuracy_diff = urva_metrics["accuracy"] - gpt_metrics["accuracy"]

    summary = {
        "urva_metrics": urva_metrics,
        "gpt_metrics": gpt_metrics,
        "hallucination_reduction": halluc_improvement,
        "conflict_score_difference": conflict_diff,
        "accuracy_difference": accuracy_diff,
    }
    return summary
