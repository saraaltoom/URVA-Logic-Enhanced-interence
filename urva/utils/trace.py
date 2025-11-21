"""
Token-flow and reasoning trace utilities.
"""
from typing import Dict, Any, List
import json
from pathlib import Path
import datetime


class TraceRecorder:
    def __init__(self, path: str | None = None):
        self.events: List[Dict[str, Any]] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, event: Dict[str, Any]) -> None:
        event = {"ts": datetime.datetime.utcnow().isoformat(), **event}
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    def export(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def clear(self) -> None:
        self.events.clear()
