import math
from typing import Dict, Any, List, Tuple
import torch
import numpy as np


class InferencePipeline:
    def __init__(self, grounder, reasoner, checker, cfg, logic):
        self.grounder = grounder
        self.reasoner = reasoner
        self.checker = checker
        self.cfg = cfg
        self.logic = logic
        self.speed_profiles = {
            "aggressive": {"refine": 0, "conflict_threshold": 0.35},
            "balanced": {"refine": 1, "conflict_threshold": 0.25},
            "deep": {"refine": 3, "conflict_threshold": 0.2},
        }

    def run(self, item: Dict[str, Any], speed: str = "balanced", debug: bool = False, ablation: str | None = None):
        profile = self.speed_profiles.get(speed, self.speed_profiles["balanced"])
        text = item["text"]
        if ablation == "refiner":
            profile = {**profile, "refine": 0}

        best_states = None
        best_graph = None
        best_logic = None
        best_conflict = 1.0

        # initial generation
        if ablation == "reasoner":
            states = {
                "S1": f"Direct: {text}",
                "S2": "",
                "S3": "",
                "hop_scores": [0.0],
                "score_tensor": torch.tensor(0.0),
                "final_score": 0.0,
            }
        else:
            states = self.reasoner({"text": text})
            if ablation == "reasoner":
                states["S1"] = f"Direct: {text}"
        graph = self._build_conflict_graph(states)
        logic_violations = [] if ablation == "logic" else self._logic_violations(states)
        best_states, best_graph, best_logic, best_conflict = states, graph, logic_violations, graph["conflict_score"]

        # refinement loop
        if ablation != "refiner" and ablation != "reasoner":
            for _ in range(profile["refine"]):
                if graph["conflict_score"] <= profile["conflict_threshold"] and not logic_violations:
                    break
                states_candidate = self.reasoner({"text": text + " (re-evaluated)"})
                graph_c = self._build_conflict_graph(states_candidate)
                logic_c = [] if ablation == "logic" else self._logic_violations(states_candidate)
                score_c = graph_c["conflict_score"] + 0.05 * len(logic_c)
                if score_c < best_conflict + 0.05 * len(best_logic):
                    best_states, best_graph, best_logic, best_conflict = (
                        states_candidate,
                        graph_c,
                        logic_c,
                        graph_c["conflict_score"],
                    )
                graph, logic_violations = graph_c, logic_c

        # chosen outputs
        states = best_states
        graph = best_graph
        logic_violations = best_logic

        if ablation == "grounder":
            grounding = {"grounded_facts": [], "avg_score": 0.0}
        else:
            grounding = self.grounder({"text": text})
        reasoning = {
            "S1": states.get("S1", ""),
            "S2": states.get("S2", ""),
            "S3": states.get("S3", ""),
            "hop_scores": states.get("hop_scores", []),
            "score_tensor": states.get("score_tensor", 0),
            "final_score": states.get("final_score", 0),
        }

        if ablation == "logic":
            halluc = {
                "violations": [],
                "has_hallucination": False,
                "type": "NONE",
                "explanation": "Logic ablated.",
            }
        else:
            halluc = self.checker.run_all(
                {"S1": reasoning["S1"], "S2": reasoning["S2"], "S3": reasoning["S3"]},
                conflict_score=graph["conflict_score"],
            )

        certainty = self._certainty(
            grounding,
            reasoning["final_score"],
            graph["conflict_score"],
            len(logic_violations),
        )

        fusion = {
            "conflict_score": graph["conflict_score"],
            "rule_violations": logic_violations,
            "reasoning_alignment": reasoning["final_score"],
            "certainty": certainty,
            "context_match": text,
        }

        natural_answer = self._naturalize(states, text, refined=profile["refine"] > 0 and (graph["conflict_score"] > profile["conflict_threshold"] or logic_violations))
        summary = self._summarize(states, text, refined=profile["refine"] > 0 and (graph["conflict_score"] > profile["conflict_threshold"] or logic_violations))
        evidence = self._evidence_line(states)

        result = {
            "id": item.get("id"),
            "final_answer": natural_answer,
            "summary": summary,
            "evidence": evidence,
            "states": states,
            "conflict_graph": graph,
            "grounding": grounding,
            "reasoning": reasoning,
            "hallucination": halluc,
            "fusion": fusion,
        }
        if debug:
            result["logic_violations"] = logic_violations
        return result

    # ----------------- Conflict Graph -----------------
    def _sentence_split(self, text: str) -> List[str]:
        parts = [p.strip() for p in text.replace("?", ".").replace("!", ".").split(".")]
        return [p for p in parts if p]

    def _embed_sentence(self, sent: str) -> torch.Tensor:
        if not sent:
            return torch.zeros(self.cfg.get("hidden_size", 128))
        rng = torch.Generator()
        rng.manual_seed(abs(hash(sent)) % (2**31 - 1))
        return torch.randn(self.cfg.get("hidden_size", 128), generator=rng)

    def _pair_relation(self, a: str, b: str) -> str:
        ea, eb = self._embed_sentence(a), self._embed_sentence(b)
        if ea.norm() == 0 or eb.norm() == 0:
            return "neutral"
        sim = torch.dot(ea, eb) / (ea.norm() * eb.norm() + 1e-8)
        sim = float(sim.detach())
        neg_tokens = {"not", "never", "no", "cannot"}
        has_neg_a = any(tok in a.lower() for tok in neg_tokens)
        has_neg_b = any(tok in b.lower() for tok in neg_tokens)
        if sim > 0.65 and has_neg_a != has_neg_b:
            return "contradiction"
        if sim > 0.7:
            return "entailment"
        if sim < 0.2 and has_neg_a != has_neg_b:
            return "contradiction"
        return "neutral"

    def _build_conflict_graph(self, states: Dict[str, Any]) -> Dict[str, Any]:
        sentences = []
        for key in ["S1", "S2", "S3"]:
            txt = states.get(key, "")
            sentences.extend(self._sentence_split(txt))

        edges = []
        contradictions = 0
        confirmations = 0
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                rel = self._pair_relation(sentences[i], sentences[j])
                edges.append({"a": sentences[i], "b": sentences[j], "type": rel})
                if rel == "contradiction":
                    contradictions += 1
                elif rel == "entailment":
                    confirmations += 1
        total = max(len(edges), 1)
        conflict_score = contradictions / total
        return {
            "contradictions": contradictions,
            "confirmations": confirmations,
            "total_relations": total,
            "conflict_score": conflict_score,
            "edges": edges,
        }

    # ----------------- Logic violations -----------------
    def _logic_violations(self, states: Dict[str, Any]) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for key in ["S1", "S2", "S3"]:
            txt = states.get(key, "")
            violations.extend(self.logic.apply_rules(txt))
        return violations

    # ----------------- Certainty -----------------
    def _certainty(
        self, grounding: Dict[str, Any], reasoning_score: float, conflict_score: float, logic_violation_count: int
    ) -> float:
        g_avg = grounding.get("avg_score", grounding.get("grounded", {}).get("avg_score", 0.5))
        penalty = min(1.0, 0.1 * logic_violation_count)
        certainty = 0.35 * g_avg + 0.35 * reasoning_score + 0.2 * (1 - conflict_score) + 0.1 * (1 - penalty)
        return float(max(0.0, min(1.0, certainty)))

    # ----------------- Natural language output -----------------
    def _clean(self, txt: str) -> str:
        return (
            txt.replace("Answer:", "")
            .replace("Explanation:", "")
            .replace("Verification:", "")
            .strip()
        )

    def _naturalize(self, states: dict, question: str, refined: bool) -> str:
        s1 = self._clean(states.get("S1", ""))
        s2 = self._clean(states.get("S2", ""))
        s3 = self._clean(states.get("S3", ""))
        for cand in [s1, s2, s3]:
            if cand and len(cand.split()) > 3:
                prefix = "Re-evaluated for consistency. " if refined else ""
                return prefix + cand
        return "The answer to your question cannot be determined."

    def _summarize(self, states: dict, question: str, refined: bool) -> str:
        s2 = self._clean(states.get("S2", ""))
        s3 = self._clean(states.get("S3", ""))
        if s2:
            base = s2
        elif s3:
            base = s3
        else:
            base = "A concise rationale was produced."
        if refined:
            return f"{base} Re-evaluated for consistency."
        return base

    def _evidence_line(self, states: dict) -> str:
        s3 = self._clean(states.get("S3", ""))
        if s3:
            return f"Evidence check: {s3}"
        return ""
