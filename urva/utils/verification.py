"""
External factual verification stub (Wikipedia/local dump hook).
"""
from typing import Dict, Any


def verify_facts(text: str) -> Dict[str, Any]:
    return {"verified": False, "source": None, "notes": "External verification not implemented (stub)."}
