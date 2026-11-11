"""
Challenger Latent Network for NetReach (v1, v2, v3, and v4 architectures).
"""

from netreach.model_v4 import ChallengerLatentNetV4
from netreach.model_v3 import ChallengerLatentNetV3
from netreach.model_v2 import ChallengerLatentNetV2
from netreach.model import ChallengerLatentNet

__all__ = [
    "ChallengerLatentNet",
    "ChallengerLatentNetV2",
    "ChallengerLatentNetV3",
    "ChallengerLatentNetV4",
]
