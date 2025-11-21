from typing import Dict, Any
from tqdm import tqdm
import torch
from torch import optim, nn
import random


class Trainer:
    """
    Minimal trainer that optimizes the reasoner and grounder to produce higher confidence
    scores. Uses BCE loss against a fixed positive target.
    """

    def __init__(self, cfg: Dict[str, Any], grounder: nn.Module, reasoner: nn.Module, checker, pipeline):
        self.cfg = cfg
        self.grounder = grounder
        self.reasoner = reasoner
        self.checker = checker
        self.pipeline = pipeline
        self.device = torch.device(cfg.get("device", "cpu"))
        self.criterion = nn.BCELoss()
        params = list(grounder.parameters()) + list(reasoner.parameters())
        self.opt = optim.Adam(params, lr=cfg.get("learning_rate", 1e-3))
        torch.manual_seed(cfg.get("seed", 42))
        random.seed(cfg.get("seed", 42))

    def run(self, loader):
        self.grounder.to(self.device)
        self.reasoner.to(self.device)
        for epoch in range(self.cfg.get("num_epochs", 1)):
            pbar = tqdm(loader.batched(), desc=f"Epoch {epoch+1}")
            for batch in pbar:
                loss = self._step(batch)
                pbar.set_postfix(loss=f"{loss:.4f}")

    def _step(self, batch):
        self.opt.zero_grad()
        total_loss = 0.0
        used = 0
        for sample in batch:
            text = sample.get("text") or sample.get("fact") or ""
            if not text:
                continue
            states = self.reasoner({"text": text})
            score = states.get("score_tensor", torch.tensor(0.5, device=self.device))
            if not isinstance(score, torch.Tensor):
                score = torch.tensor(score, device=self.device)
            target = torch.ones_like(score)
            loss = self.criterion(score, target)
            loss.backward()
            total_loss += loss.item()
            used += 1
        self.opt.step()
        return total_loss / max(used, 1)
