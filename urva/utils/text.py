from typing import List


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in text.replace("!", ".").replace("?", ".").split(".")]
    return [p for p in parts if p]
