#  Copyright (c) 2024-2026. KU Leuven
"""Lower a Circuit to a DeepLogModule, via Klay or the generic evaluator.

The last stage of compiling a formula: a circuit in, a runnable torch module
out. Distinct from :func:`~deeplog.circuit.knowledge_compile`, an earlier
pass over circuits — a circuit in, an equivalent d-DNNF circuit out.
"""
