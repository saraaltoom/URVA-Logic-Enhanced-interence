import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime


class JsonLogger:
    """
    Simple JSONL logger for tracing events (token-flow, rule violations, etc.).
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = {"timestamp": datetime.datetime.utcnow().isoformat(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class TraceBuffer:
    """
    In-memory trace collector used for token-flow and reasoning traces.
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def export(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()
