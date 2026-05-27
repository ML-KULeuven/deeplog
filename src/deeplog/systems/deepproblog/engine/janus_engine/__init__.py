"""
A module providing the Janus-based Prolog engine.
"""

#  Copyright (c) 2024-2026. KU Leuven

from .janus_engine import JANUS_AVAILABLE
from .janus_engine import JanusEngine
from .janus_engine import JanusNotAvailableException


__all__ = [
    "JANUS_AVAILABLE",
    "JanusEngine",
    "JanusNotAvailableException",
]
