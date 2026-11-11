"""
NetReach: Enterprise Neural Attack Path Reachability & Lateral Movement Analysis Engine.
"""

from netreach.model import ChallengerLatentNet
from netreach.model_v2 import ChallengerLatentNetV2
from netreach.model_v3 import ChallengerLatentNetV3
from netreach.model_v4 import ChallengerLatentNetV4, RELATION_TYPES, RELATION_TO_ID
from netreach.engine import NetReachEngine, AuditResult
from netreach.adapters.graph_loader import load_graph
from netreach.adapters.extractor import extract_subgraph, prepare_graph_batch
from netreach.adapters.bloodhound import (
    load_bloodhound_graph,
    parse_bloodhound_data,
    normalize_ad_relation,
    AD_RELATION_MAP,
)
from netreach.export import quantize_model_for_cpu, export_quantized_checkpoint

__version__ = "0.5.0"

__all__ = [
    "NetReachEngine",
    "AuditResult",
    "ChallengerLatentNet",
    "ChallengerLatentNetV2",
    "ChallengerLatentNetV3",
    "ChallengerLatentNetV4",
    "RELATION_TYPES",
    "RELATION_TO_ID",
    "load_graph",
    "extract_subgraph",
    "prepare_graph_batch",
    "load_bloodhound_graph",
    "parse_bloodhound_data",
    "normalize_ad_relation",
    "AD_RELATION_MAP",
    "quantize_model_for_cpu",
    "export_quantized_checkpoint",
    "__version__",
]
