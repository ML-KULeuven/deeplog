#  Copyright (c) 2024-2026. KU Leuven
"""The SWI-Prolog (Janus) grounder, and the prover it runs on."""

from .janus import JanusGrounder as JanusGrounder
from .prover import JANUS_AVAILABLE as JANUS_AVAILABLE
from .prover import JanusNotAvailableException as JanusNotAvailableException
from .prover import JanusProver as JanusProver
