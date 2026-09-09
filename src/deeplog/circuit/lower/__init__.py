#  Copyright (c) 2024-2026. KU Leuven
"""Lower a Circuit to a DeepLogModule, via Klay or the generic evaluator.

The end of the pipeline: a circuit in, a runnable torch module out. Distinct
from :mod:`deeplog.circuit.knowledge_compile`, which is a *pass* within it —
a circuit in, an equivalent d-DNNF circuit out.
"""
