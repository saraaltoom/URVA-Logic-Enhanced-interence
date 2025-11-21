"""
Mixed precision autocast wrapper.
"""
from contextlib import contextmanager
import torch


@contextmanager
def maybe_autocast(enabled: bool = True):
    with torch.cuda.amp.autocast(enabled=enabled):
        yield
