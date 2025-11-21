"""
Scalable training loop with gradient accumulation, mixed precision, checkpointing, and early stopping.
"""
from typing import Dict, Any
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.cuda.amp import GradScaler, autocast
import random
import os

from urva.utils.seed import set_seed
from urva.core.optim import build_optimizer
from urva.core.schedulers import build_scheduler
from urva.core.checkpoint import save_checkpoint
from urva.core.amp import maybe_autocast


class Trainer:
    def __init__(self, cfg: Dict[str, Any], grounder: nn.Module, reasoner: nn.Module, checker, pipeline):
        self.cfg = cfg
        self.grounder = grounder
        self.reasoner = reasoner
        self.checker = checker
        self.pipeline = pipeline
        self.device = torch.device(cfg.get("device", "cpu"))
        self.criterion = nn.BCELoss()
        params = list(grounder.parameters()) + list(reasoner.parameters())
        self.opt = build_optimizer(params, cfg)
        self.scheduler = build_scheduler(self.opt, cfg)
        self.scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
        self.grad_accum = cfg.get("grad_accum_steps", 1)
        self.eval_interval = cfg.get("eval_interval", 100)
        self.checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        set_seed(cfg.get("seed", 42))

    def _checkpoint(self, step: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"ckpt_{step}.pt")
        save_checkpoint(
            path,
            {
                "grounder": self.grounder.state_dict(),
                "reasoner": self.reasoner.state_dict(),
                "optimizer": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": step,
            },
        )

    def run(self, loader):
        self.grounder.to(self.device)
        self.reasoner.to(self.device)
        step = 0
        best_loss = float("inf")
        for epoch in range(self.cfg.get("num_epochs", 1)):
            pbar = tqdm(loader.batched(), desc=f"Epoch {epoch+1}")
            for batch in pbar:
                loss = self._step(batch)
                step += 1
                pbar.set_postfix(loss=f"{loss:.4f}")
                if self.scheduler:
                    self.scheduler.step()
                if step % self.eval_interval == 0:
                    self._checkpoint(step)
                if loss < best_loss:
                    best_loss = loss
            if best_loss < 1e-3:
                break

    def _step(self, batch):
        self.opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        used = 0
        for micro_idx in range(len(batch)):
            sample = batch[micro_idx]
            text = sample.get("text") or sample.get("fact") or ""
            if not text:
                continue
            states = self.reasoner({"text": text})
            score = states.get("score_tensor", torch.tensor(0.5, device=self.device))
            if not isinstance(score, torch.Tensor):
                score = torch.tensor(score, device=self.device)
            target = torch.ones_like(score)
            with maybe_autocast(enabled=self.cfg.get("mixed_precision", True)):
                loss = self.criterion(score, target)
            self.scaler.scale(loss / self.grad_accum).backward()
            total_loss += loss.item()
            used += 1
            if (micro_idx + 1) % self.grad_accum == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
        if used and (len(batch) % self.grad_accum) != 0:
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
        return total_loss / max(used, 1)
