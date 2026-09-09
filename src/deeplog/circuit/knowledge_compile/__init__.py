#  Copyright (c) 2024-2026. KU Leuven
"""Knowledge compilation: a circuit in, an equivalent d-DNNF circuit out.

The input is an arbitrary formula over an algebra; the output is an equivalent
circuit that is **deterministic** (a disjunction's branches are mutually
exclusive) and **decomposable** (a conjunction's operands share no variables).

Those are the properties under which
:func:`~deeplog.circuit.transform.transform_circuit` is exact — reading ``or``
as a sum would claim ``P(a) + P(b)`` for ``a ∨ b`` in general, and is correct on
a d-DNNF, where the branches cannot both hold. A weighted model count is
therefore a knowledge compilation followed by a transform.

The output is an ordinary circuit, so any evaluator can take it — this is a
pass over circuits, not the lowering to a torch module, which is
:mod:`deeplog.circuit.lower`.
"""

from .dispatch import knowledge_compile


__all__ = ["knowledge_compile"]
