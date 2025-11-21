"""
Graph visualization stubs (export edges/nodes for external tools).
"""
from typing import Dict, Any, List


def export_conflict_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "nodes": list({n for e in graph.get("edges", []) for n in [e.get("a"), e.get("b")] if n}),
        "edges": graph.get("edges", []),
        "conflict_score": graph.get("conflict_score", 0.0),
        "spectral": graph.get("spectral", 0.0),
    }
