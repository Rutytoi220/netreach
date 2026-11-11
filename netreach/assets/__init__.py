import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(ASSETS_DIR, "weights.pt")
DEFAULT_V4_WEIGHTS_PATH = os.path.join(ASSETS_DIR, "weights_v4.pt")

__all__ = ["ASSETS_DIR", "DEFAULT_WEIGHTS_PATH", "DEFAULT_V4_WEIGHTS_PATH"]
