import argparse
from urva.config import load_config
from urva.data.loader import DatasetLoader
from urva.logic.engine import LogicEngine
from urva.models.grounder import FactGrounder
from urva.models.reasoner import MultiHopReasoner
from urva.checks.hallucination import HallucinationChecker
from urva.pipeline.inference import InferencePipeline
from urva.train.training_loop import Trainer
from urva.eval.evaluate import Evaluator
from urva.eval.metrics import compute_metrics, summarize
from urva.data import benchmarks
from urva.eval.baseline_compare import compare_urva_vs_gpt


def format_output(out):
    lines = [
        f"Final Answer: {out.get('final_answer','')}",
        f"Summary: {out.get('summary','')}",
        f"Certainty: {out['fusion'].get('certainty',0):.3f}",
        f"Reasoning Confidence: {out['fusion'].get('reasoning_alignment',0):.3f}",
        f"Conflict Score: {out['fusion'].get('conflict_score',0):.3f}",
        f"Logic Violations: {len(out.get('hallucination',{}).get('violations',[]))}",
        f"Hallucination Type: {out.get('hallucination',{}).get('type','NONE')}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="URVA Beast-Mode CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "infer", "bench", "baseline"], default="infer")
    parser.add_argument("--speed", type=str, choices=["aggressive", "balanced", "deep"], default="balanced")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file (jsonl or json array)")
    parser.add_argument("--benchmark", type=str, choices=["truthfulqa_mc", "truthfulqa_gen", "hotpot"], help="Benchmark selection for bench/baseline modes")
    parser.add_argument("--logic", type=str, default="logic_rules.json", help="Path to logic rules JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, help="Ad-hoc inference text")
    parser.add_argument("--debug", action="store_true", help="Include debug tensors/objects")
    parser.add_argument("--ablation", type=str, choices=["grounder", "reasoner", "logic", "refiner"], help="Remove a component for ablation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    loader = DatasetLoader(args.data, cfg)
    logic = LogicEngine.from_file(args.logic)
    grounder = FactGrounder(cfg)
    reasoner = MultiHopReasoner(cfg)
    checker = HallucinationChecker(logic, conflict_threshold=cfg.get("graph", {}).get("conflict_threshold", 0.25))
    pipeline = InferencePipeline(grounder, reasoner, checker, cfg, logic)
    if args.ablation:
        print(f"Ablation active: {args.ablation} removed")

    if args.mode == "train":
        trainer = Trainer(cfg, grounder, reasoner, checker, pipeline)
        trainer.run(loader)
    elif args.mode == "eval":
        evaluator = Evaluator(cfg, pipeline)
        evaluator.run(loader)
    elif args.mode == "bench":
        if not args.benchmark:
            raise SystemExit("Specify --benchmark for bench mode")
        if args.benchmark == "truthfulqa_mc":
            dataset = benchmarks.load_truthfulqa_mc(args.data)
        elif args.benchmark == "truthfulqa_gen":
            dataset = benchmarks.load_truthfulqa_gen(args.data)
        else:
            dataset = benchmarks.load_hotpot(args.data)
        outputs = [pipeline.run(sample, speed=args.speed, ablation=args.ablation) for sample in dataset]
        metrics = compute_metrics(outputs)
        print(summarize(metrics))
    elif args.mode == "baseline":
        if not args.benchmark:
            raise SystemExit("Specify --benchmark for baseline mode")
        if args.benchmark == "truthfulqa_mc":
            dataset = benchmarks.load_truthfulqa_mc(args.data)
        elif args.benchmark == "truthfulqa_gen":
            dataset = benchmarks.load_truthfulqa_gen(args.data)
        else:
            dataset = benchmarks.load_hotpot(args.data)
        summary = compare_urva_vs_gpt(dataset, pipeline, logic, cfg, speed=args.speed, ablation=args.ablation)
        urva_acc = summary["urva_metrics"]["accuracy"]
        gpt_acc = summary["gpt_metrics"]["accuracy"]
        print(
            f"URVA accuracy: {urva_acc:.3f} | GPT accuracy: {gpt_acc:.3f} | "
            f"Hallucination reduction: {summary['hallucination_reduction']:.3f} | "
            f"Conflict diff: {summary['conflict_score_difference']:.3f} | "
            f"Accuracy diff: {summary['accuracy_difference']:.3f}"
        )
    else:
        if args.text:
            sample = {"id": "adhoc", "text": args.text}
            output = pipeline.run(sample, speed=args.speed, debug=args.debug, ablation=args.ablation)
            print(format_output(output))
        else:
            for sample in loader:
                out = pipeline.run(sample, speed=args.speed, debug=args.debug, ablation=args.ablation)
                print(format_output(out))


if __name__ == "__main__":
    main()
