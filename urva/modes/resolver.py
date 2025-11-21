from dataclasses import dataclass


@dataclass
class ModeConfig:
    speed: str
    build_graph: bool
    full_rules: bool
    refine_steps: int
    multi_state: bool


def resolve_mode(speed: str, cfg) -> ModeConfig:
    speed = speed.lower()
    if speed not in ("aggressive", "smart", "turbo"):
        speed = "smart"
    if speed == "aggressive":
        return ModeConfig(speed, build_graph=True, full_rules=True, refine_steps=cfg["refine_loops"]["aggressive"], multi_state=True)
    if speed == "smart":
        return ModeConfig(speed, build_graph=True, full_rules=False, refine_steps=cfg["refine_loops"]["smart"], multi_state=True)
    return ModeConfig(speed, build_graph=False, full_rules=False, refine_steps=cfg["refine_loops"]["turbo"], multi_state=False)
