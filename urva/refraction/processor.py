from typing import Dict, Any


class RefractionLayer:
    """
    Produces refracted inputs: Q_raw, Q_logic, Q_context.
    """

    def __init__(self, logic_engine):
        self.logic_engine = logic_engine

    def run(self, sample: Dict[str, Any]) -> Dict[str, str]:
        text = sample.get("text") or sample.get("fact") or ""
        q_raw = text.strip()
        q_logic = f"[LOGIC_VIEW] {q_raw}"
        q_context = sample.get("context", "") or q_raw
        return {"Q_raw": q_raw, "Q_logic": q_logic, "Q_context": q_context}
