"""Reference-resolution helpers for Sphinx and AutoAPI."""

from __future__ import annotations


#  Copyright (c) 2024-2026. KU Leuven

_REF_TARGET_ALIASES = {
    "deeplog.module.predicate_modules.builtin_predicates.EqualityPredicate": (
        "deeplog.formula.predicates.builtin_predicates.EqualityPredicate"
    ),
    "deeplog.module.predicate_modules.builtin_predicates.ProbabilityPredicate": (
        "deeplog.formula.predicates.builtin_predicates.ProbabilityPredicate"
    ),
    "deeplog.module.predicate_modules.builtin_predicates.SumsPredicate": (
        "deeplog.formula.predicates.builtin_predicates.SumsPredicate"
    ),
    "deeplog.module.predicate_modules.builtin_predicates.get_network_predicate": (
        "deeplog.formula.predicates.builtin_predicates.get_network_predicate"
    ),
}


def resolve_autoapi_xref(app, env, node, contnode):
    """Resolve a handful of AutoAPI cross-reference mismatches."""
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    if not target:
        return None
    target = _REF_TARGET_ALIASES.get(target, target)

    domain = env.get_domain("py")
    refdoc = node.get("refdoc", "")
    objects = env.domaindata["py"]["objects"]

    if target in objects:
        return domain.resolve_xref(
            env, refdoc, app.builder, "obj", target, node, contnode
        )

    name = target.rsplit(".", 1)[-1]
    prefix = target.rsplit(".", 1)[0] + "." if "." in target else ""
    candidates = [
        full
        for full in objects
        if full.rsplit(".", 1)[-1] == name and (not prefix or full.startswith(prefix))
    ]
    if len(candidates) == 1:
        return domain.resolve_xref(
            env, refdoc, app.builder, "obj", candidates[0], node, contnode
        )
    return None
