import json
from typing import List, Dict, Any
import pathlib


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    if "\n" in text.strip().splitlines()[0]:
        return []
    try:
        if text.strip().startswith("["):
            return json.loads(text)
    except Exception:
        pass
    # fallback jsonl
    data = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return data


def _normalize(entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
    q = entry.get("question") or entry.get("query") or entry.get("prompt") or entry.get("text")
    if not q and "input" in entry:
        q = entry["input"]
    ans = entry.get("answer") or entry.get("best_answer") or entry.get("target") or entry.get("reference")
    return {
        "id": entry.get("id", idx),
        "text": q or "",
        "answer": ans or "",
    }


def load_truthfulqa_mc(path: str) -> List[Dict[str, Any]]:
    raw = _load_json_or_jsonl(path)
    normalized = []
    for idx, e in enumerate(raw):
        n = _normalize(e, idx)
        normalized.append(n)
    return normalized


def load_truthfulqa_gen(path: str) -> List[Dict[str, Any]]:
    raw = _load_json_or_jsonl(path)
    normalized = []
    for idx, e in enumerate(raw):
        n = _normalize(e, idx)
        normalized.append(n)
    return normalized


def load_hotpot(path: str) -> List[Dict[str, Any]]:
    raw = _load_json_or_jsonl(path)
    normalized = []
    for idx, e in enumerate(raw):
        q = e.get("question") or e.get("query") or e.get("text") or e.get("prompt")
        ans = e.get("answer") or e.get("reference") or e.get("label")
        normalized.append({"id": e.get("id", idx), "text": q or "", "answer": ans or ""})
    return normalized
