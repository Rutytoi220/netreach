"""
Challenger Latent Network for NetReach (v1, v2, and v3 architectures).
"""

from netreach.model_v3 import ChallengerLatentNetV3
from netreach.model_v2 import ChallengerLatentNetV2
from netreach.model import ChallengerLatentNet

__all__ = [
    "ChallengerLatentNet",
    "ChallengerLatentNetV2",
    "ChallengerLatentNetV3",
]
