import torch


def get_device(name: str) -> torch.device:
    return torch.device(name if torch.cuda.is_available() or "cuda" not in name else "cpu")
