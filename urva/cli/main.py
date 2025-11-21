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


def main():
    parser = argparse.ArgumentParser(description="URVA Architecture CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "infer"], default="infer")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file (jsonl)")
    parser.add_argument("--logic", type=str, default="logic_rules.json", help="Path to logic rules JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, help="Ad-hoc inference text")
    args = parser.parse_args()

    cfg = load_config(args.config)
    loader = DatasetLoader(args.data, cfg)
    logic = LogicEngine.from_file(args.logic)
    grounder = FactGrounder(cfg)
    reasoner = MultiHopReasoner(cfg)
    checker = HallucinationChecker(logic)
    pipeline = InferencePipeline(grounder, reasoner, checker, cfg)

    if args.mode == "train":
        trainer = Trainer(cfg, grounder, reasoner, checker, pipeline)
        trainer.run(loader)
    elif args.mode == "eval":
        evaluator = Evaluator(cfg, pipeline)
        evaluator.run(loader)
    else:
        if not args.text:
            raise SystemExit("Provide --text for inference mode")
        output = pipeline.run({"id": "adhoc", "text": args.text})
        print(output)


if __name__ == "__main__":
    main()
