"""
NetReach: Neural Attack Path Reachability & Lateral Movement Analysis Engine.
"""

from netreach.model import ChallengerLatentNet
from netreach.model_v2 import ChallengerLatentNetV2, RELATION_TYPES, RELATION_TO_ID
from netreach.engine import NetReachEngine, AuditResult
from netreach.adapters.graph_loader import load_graph
from netreach.adapters.extractor import extract_subgraph

__version__ = "0.2.0"

__all__ = [
    "NetReachEngine",
    "AuditResult",
    "ChallengerLatentNet",
    "ChallengerLatentNetV2",
    "RELATION_TYPES",
    "RELATION_TO_ID",
    "load_graph",
    "extract_subgraph",
    "__version__",
]
