#  Copyright (c) 2024-2026. KU Leuven
"""MV-SDD compile backend with constant-baking.

Used for deterministic circuits with annotated-disjunction (categorical)
leaves. One multi-valued variable per AD; numeric-constant labels are
baked into the AC so only neural-network-backed slots remain runtime inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import klay

from ...module.wrappers import ConstantPrefillModule
from ...module.wrappers import WrappedModule
from ...shape import SymTensor
from .common import klay_to_torch_module
from .common import walk_and_combine


if TYPE_CHECKING:
    from ...module import DeepLogModule
    from ...symbol import Symbol
    from ..circuit import Circuit


def compile_mvsdd(
    circuit: Circuit,
    roots: dict[int, Symbol],
    structure_override: str | None,
    categoricals: dict[Symbol, tuple[str, int | str]],
    labels: dict[Symbol, Symbol],
) -> DeepLogModule:
    """Compile a circuit to a torch module via MV-SDD canonicalisation.

    Build steps:
        1. Allocate one MV-SDD variable per categorical (cat_id) and one
           per non-categorical boolean leaf. Each (mv_var, mv_value) pair
           gets a unique positive klay literal id (one-hot indicator
           slot).
        2. Build the MV-SDD bottom-up by traversing the circuit's DAG
           and applying the corresponding mvsdd_* operation per node.
        3. Walk the resulting (canonical, mutex-respecting) MV-SDD via
           the visitor protocol on ``Node`` and emit klay primitives —
           terminals expand to OR-of-indicators, decomp pairs to nested
           AND/OR.
        4. Hand the klay circuit to ``klay.to_torch_module``, exactly
           the same call the PySDD path makes.

    The torch module sees one input slot per (mv_var, mv_value) pair.
    Boolean leaves take 2 slots each (value=0 / value=1); a categorical
    with N branches plus a "none" outcome takes N+1 slots, fed by a
    one-hot from upstream softmax (deeplog wrapper's responsibility).
    """
    import pymvsdd

    # 1a. Identify which leaves belong to which MV variable.
    #
    # Each leaf's name is ("_", goal, (structure,)). For categorical
    # leaves the goal appears in the categoricals map; group those by
    # cat_id. For "none" outcomes the goal is ("@cat_none", (cat_id,)) —
    # already keyed in categoricals with value="none".
    leaf_nodes = circuit.leaf_nodes  # {leaf_symbol: node_id}

    cat_id_to_leaves: dict[str, list[tuple[Symbol, int]]] = {}
    boolean_leaves: list[tuple[Symbol, int]] = []
    for leaf_symbol, leaf_id in leaf_nodes.items():
        # leaf_symbol = ("_", goal, (structure,)); goal is the inner Symbol.
        goal = leaf_symbol[1] if len(leaf_symbol) >= 2 else leaf_symbol
        if goal in categoricals:
            cat_id, _value = categoricals[goal]
            cat_id_to_leaves.setdefault(cat_id, []).append((leaf_symbol, leaf_id))
        else:
            boolean_leaves.append((leaf_symbol, leaf_id))

    # 1b. Allocate MV variables. Variable order is deterministic (cat_ids
    # by sort order, then boolean leaves in their iteration order) so
    # repeated compilation of the same circuit produces identical vtrees.
    cat_id_order: list[str] = sorted(cat_id_to_leaves)
    bool_leaf_order: list[Symbol] = [leaf for leaf, _ in boolean_leaves]

    mv_var_of_cat_id: dict[str, int] = {}
    domain_sizes: list[int] = []
    next_mv_var = 0

    def _cat_values(cat_id: str) -> list:
        """Return the (cat_id, value) value-fields for every leaf in this cat."""
        # ls is a Symbol; ls[1] is its first child Symbol. Pyright can't
        # narrow tuple[str, *tuple[Symbol, ...]] indexing past index 0.
        return [
            categoricals[cast("Symbol", ls[1])][1] for ls, _ in cat_id_to_leaves[cat_id]
        ]

    for cat_id in cat_id_order:
        # Domain size = max value index + 1 across this cat_id's leaves.
        # Numeric values are AD-branch indices; the literal "none" gets
        # the next index after the largest numeric one. mv-sdd requires
        # domain_size >= 2 — pad to 2 for the (rare) single-branch case.
        values = _cat_values(cat_id)
        numeric = [v for v in values if isinstance(v, int)]
        has_none = any(v == "none" for v in values)
        max_idx = max(numeric) if numeric else -1
        size = max_idx + 1 + (1 if has_none else 0)
        domain_sizes.append(max(size, 2))
        mv_var_of_cat_id[cat_id] = next_mv_var
        next_mv_var += 1

    mv_var_of_bool_leaf: dict[Symbol, int] = {}
    for leaf_symbol in bool_leaf_order:
        mv_var_of_bool_leaf[leaf_symbol] = next_mv_var
        domain_sizes.append(2)
        next_mv_var += 1

    if not domain_sizes:
        # Edge case: a circuit with only constant nodes. Pad with a
        # dummy variable so mv-sdd accepts the vtree, then ignore.
        domain_sizes = [2]

    # 2. Build the MV-SDD bottom-up by traversing the deeplog Circuit.
    vtree = pymvsdd.Vtree.right_linear(domain_sizes)
    mgr = pymvsdd.Manager(vtree)

    node_to_mv: dict[int, pymvsdd.Node] = {}

    # Pre-fill leaves.
    for cat_id, leaves in cat_id_to_leaves.items():
        var = mv_var_of_cat_id[cat_id]
        # Pre-compute the "none" outcome's value-idx (max numeric idx + 1).
        numeric = [v for v in _cat_values(cat_id) if isinstance(v, int)]
        none_idx = (max(numeric) + 1) if numeric else 0
        for leaf_symbol, leaf_id in leaves:
            goal = cast("Symbol", leaf_symbol[1])
            _cid, value = categoricals[goal]
            value_idx = none_idx if value == "none" else int(value)
            node_to_mv[leaf_id] = mgr.literal(var, value_idx)

    for leaf_symbol, leaf_id in boolean_leaves:
        var = mv_var_of_bool_leaf[leaf_symbol]
        # Convention: value=1 means "true", value=0 means "false".
        node_to_mv[leaf_id] = mgr.literal(var, 1)

    # Map deeplog constant nodes to MV-SDD constants.
    zero_id = circuit.zero_node
    one_id = circuit.one_node
    if zero_id is not None:
        node_to_mv[zero_id] = mgr.bot()
    if one_id is not None:
        node_to_mv[one_id] = mgr.top()

    # Walk and build internal nodes.
    walk_and_combine(circuit, roots, node_to_mv)  # pyright: ignore[reportArgumentType]

    # 3. Allocate klay literal ids per (mv_var, mv_value) pair and walk
    # the canonicalised MV-SDD into a klay circuit.
    klay_circuit = klay.Circuit()
    next_lit_id = 1
    lit_id_for: dict[tuple[int, int], int] = {}

    def get_lit_id(var: int, value: int) -> int:
        nonlocal next_lit_id
        key = (var, value)
        if key not in lit_id_for:
            lit_id_for[key] = next_lit_id
            next_lit_id += 1
        return lit_id_for[key]

    # Pre-allocate a klay literal id for every input slot we expect to
    # see. This keeps the klay leaf-id space dense even if some
    # (var, value) pairs never appear in any reachable terminal.
    for cat_id in cat_id_order:
        var = mv_var_of_cat_id[cat_id]
        for v in range(domain_sizes[var]):
            get_lit_id(var, v)
    for leaf_symbol in bool_leaf_order:
        var = mv_var_of_bool_leaf[leaf_symbol]
        get_lit_id(var, 0)
        get_lit_id(var, 1)

    def emit(node):
        # Memoise on the canonical mv-sdd node id, NOT Python id() —
        # `node.elements` allocates fresh Node wrappers on every call,
        # so two Python wrappers around the same canonical C node have
        # different id()s. Without this, shared subdiagrams (which is
        # the whole point of an SDD) get re-emitted into klay as
        # disjoint copies and arithmetic-circuit semantics break.
        nid = node.node_id
        cached = emit_memo.get(nid)
        if cached is not None:
            return cached
        kind = node.kind
        if kind == "top":
            r = klay_circuit.true_node()
        elif kind == "bot":
            r = klay_circuit.false_node()
        elif kind == "terminal":
            var = node.var
            indicators = [
                klay_circuit.literal_node(get_lit_id(var, v)) for v in node.values
            ]
            r = (
                indicators[0]
                if len(indicators) == 1
                else klay_circuit.or_node(indicators)
            )
        elif kind == "decomp":
            elements = [
                klay_circuit.and_node([emit(prime), emit(sub)])
                for prime, sub in node.elements
            ]
            r = elements[0] if len(elements) == 1 else klay_circuit.or_node(elements)
        else:
            raise ValueError(f"Unknown MV-SDD node kind: {kind}")
        emit_memo[nid] = r
        return r

    emit_memo: dict[int, klay.NodePtr] = {}
    for root_id in roots:
        klay_circuit.set_root(emit(node_to_mv[root_id]))

    # 4. Convert klay circuit to torch module.
    torch_module = klay_to_torch_module(
        klay_circuit,
        semiring_key=structure_override or circuit.structure,
    )

    # The module's input is one slot per (mv_var, mv_value) pair, in
    # klay-literal-id order (1..N). Surface the corresponding leaf
    # symbols so callers can shape inputs correctly.
    input_symbols: list[Symbol] = [None] * (next_lit_id - 1)  # type: ignore[list-item]
    # Categorical slots: name them ("@cat", cat_id, value_str)
    for cat_id in cat_id_order:
        var = mv_var_of_cat_id[cat_id]
        for v in range(domain_sizes[var]):
            slot = lit_id_for[(var, v)] - 1
            input_symbols[slot] = ("@cat", (cat_id,), (str(v),))
    # Boolean slots: name them as ("_", goal, ("boolean",)) for value=1
    # (the original leaf) and ("_", goal, ("boolean", "false")) for value=0.
    for leaf_symbol in bool_leaf_order:
        var = mv_var_of_bool_leaf[leaf_symbol]
        slot_true = lit_id_for[(var, 1)] - 1
        slot_false = lit_id_for[(var, 0)] - 1
        input_symbols[slot_true] = leaf_symbol
        # Distinguish the false-indicator slot with an explicit tag.
        false_tag: Symbol = ("@bool_false", cast("Symbol", leaf_symbol[1]))
        input_symbols[slot_false] = false_tag

    output_symbols = [
        (
            (sym[0], sym[1], (structure_override,))
            if structure_override and len(sym) == 3
            else sym
        )
        for sym in roots.values()
    ]

    # Constant-baking: any input slot whose backing label is a numeric
    # constant gets pre-filled by a thin wrapper. Only slots with
    # non-numeric labels (typically neural-network-backed predicates)
    # remain visible as runtime inputs.
    structure = circuit.algebraic_structure
    n_total_slots = next_lit_id - 1
    constant_slot_values: dict[int, float] = {}
    free_input_symbols: list[Symbol] = []
    free_slot_indices: list[int] = []

    # Map each cat (cat_id, value_or_none) to its goal Symbol so we can
    # look the label up. Built from `categoricals` reversed; "none"
    # outcomes have a synthetic goal under @cat_none.
    cat_to_goal: dict[tuple[str, int | str], Symbol] = {
        v: g for g, v in categoricals.items()
    }
    none_value_idx_for_cat: dict[str, int] = {}
    for cat_id in cat_id_order:
        cat_values = _cat_values(cat_id)
        numeric = [v for v in cat_values if isinstance(v, int)]
        if any(v == "none" for v in cat_values):
            none_value_idx_for_cat[cat_id] = (max(numeric) + 1) if numeric else 0

    for slot_idx in range(n_total_slots):
        sym = input_symbols[slot_idx]
        const_value: float | None = None
        if sym[0] == "@cat":
            cat_id = sym[1][0]
            v = int(sym[2][0])
            none_idx = none_value_idx_for_cat.get(cat_id)
            cat_value: int | str = "none" if v == none_idx else v
            goal = cat_to_goal.get((cat_id, cat_value))
            if goal is None:
                # Padding slot — no leaf maps here. Bake as 0.
                const_value = 0.0
            else:
                label = labels.get(goal)
                if label is not None:
                    const_value = structure.get_constant_value(label)
        # Boolean indicator slots stay free — there's no label-based
        # constant to substitute (the user's runtime input is the
        # truth indicator). The "@bool_false" companion slot will
        # similarly be a free input.
        if const_value is not None:
            constant_slot_values[slot_idx] = float(const_value)
        else:
            free_slot_indices.append(slot_idx)
            free_input_symbols.append(sym)

    if not constant_slot_values:
        # No constants to bake — the klay module is the wrapper as-is.
        return WrappedModule(
            torch_module,
            SymTensor(input_symbols),
            SymTensor(output_symbols),
            name=circuit.name,
            vmap=True,
        )

    return WrappedModule(
        ConstantPrefillModule(
            torch_module,
            n_total_slots=n_total_slots,
            free_slot_indices=tuple(free_slot_indices),
            constant_slot_values=constant_slot_values,
        ),
        SymTensor(free_input_symbols),
        SymTensor(output_symbols),
        name=circuit.name,
        vmap=True,
    )
