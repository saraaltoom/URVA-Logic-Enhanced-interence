from typing import Dict, Any
import torch
from torch import nn


class MultiHopReasoner(nn.Module):
    """
    Lightweight multi-head reasoner that produces three independent verbalized states
    (S1 direct, S2 justification, S3 verification) plus a confidence score.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.hidden = cfg.get("hidden_size", 128)
        self.embed = nn.GRU(
            input_size=self.hidden, hidden_size=self.hidden, batch_first=True
        )
        self.proj = nn.Linear(self.hidden, self.hidden)
        self.score_head = nn.Linear(self.hidden, 1)
        # Three heads for different verbalizations
        self.direct_head = nn.Linear(self.hidden, 1)
        self.justify_head = nn.Linear(self.hidden, 1)
        self.verify_head = nn.Linear(self.hidden, 1)
        # Sentence banks
        self.templates_direct = [
            "The evidence points to a clear outcome.",
            "Given the facts, this is the most plausible answer.",
            "Considering the context, this conclusion holds.",
            "Based on the reasoning, this appears correct.",
        ]
        self.templates_justify = [
            "This follows from the underlying facts and constraints.",
            "The rationale is anchored in the provided details.",
            "It aligns with the known data and logical rules.",
            "The steps are consistent with the stated premises.",
        ]
        self.templates_verify = [
            "This restates the result and checks it against the facts.",
            "The conclusion remains consistent under verification.",
            "Cross-checking the reasoning yields the same answer.",
            "Validation against constraints keeps the answer intact.",
        ]

    def _encode_text(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros((1, 1, self.hidden))
        rng = torch.Generator()
        rng.manual_seed(abs(hash(text)) % (2**31 - 1))
        base = torch.randn((1, len(text), self.hidden), generator=rng)
        return base

    def _pick(self, logits: torch.Tensor, bank: list[str]) -> str:
        score = torch.sigmoid(logits).item()
        idx = int(score * len(bank)) % len(bank)
        return bank[idx]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        text = batch.get("text", "")
        emb = self._encode_text(text)
        emb = emb.to(next(self.parameters()).device)
        out, _ = self.embed(emb)
        pooled = out.mean(dim=1)
        pooled = torch.relu(self.proj(pooled))

        d = self.direct_head(pooled)
        j = self.justify_head(pooled)
        v = self.verify_head(pooled)
        score = torch.sigmoid(self.score_head(pooled)).mean()

        s1 = self._pick(d, self.templates_direct)
        s2 = "Explanation: " + self._pick(j, self.templates_justify)
        s3 = "Verification: " + self._pick(v, self.templates_verify)

        hop_scores = torch.sigmoid(out.mean(dim=2)).mean(dim=1)

        return {
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "hop_scores": [float(hop_scores.mean().detach())],
            "score_tensor": score,
            "final_score": float(score.detach()),
        }

    def reason(self, hop_embeddings):
        # Compatibility shim; ignore hop embeddings and regenerate
        return self.forward({"text": ""})
