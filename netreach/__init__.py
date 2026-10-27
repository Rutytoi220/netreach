"""
NetReach: Neural Attack Path Reachability & Lateral Movement Analysis Engine.
"""

from netreach.model import ChallengerLatentNet
from netreach.engine import NetReachEngine, AuditResult
from netreach.adapters.graph_loader import load_graph
from netreach.adapters.extractor import extract_subgraph

__version__ = "0.1.0"

__all__ = [
    "NetReachEngine",
    "AuditResult",
    "ChallengerLatentNet",
    "load_graph",
    "extract_subgraph",
    "__version__",
]
