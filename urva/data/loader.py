import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List


class DatasetLoader:
    def __init__(self, path: str, cfg: Dict[str, Any]):
        self.path = Path(path)
        self.cfg = cfg
        random.seed(cfg["seed"])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Support both JSONL and JSON array files.
        with self.path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return
            if content.startswith("["):
                try:
                    items = json.loads(content)
                    if isinstance(items, dict):
                        items = [items]
                    for item in items:
                        yield item
                    return
                except json.JSONDecodeError:
                    # fall back to line-by-line parsing
                    f.seek(0)
                    content = f.read()
            # line-delimited JSON
            for line in content.splitlines():
                if not line.strip():
                    continue
                yield json.loads(line)

    def batched(self, batch_size: int | None = None) -> Iterator[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        bsz = batch_size or self.cfg["batch_size"]
        for item in self:
            batch.append(item)
            if len(batch) >= bsz:
                yield batch
                batch = []
        if batch:
            yield batch
