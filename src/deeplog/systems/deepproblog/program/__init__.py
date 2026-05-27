#  Copyright (c) 2024-2026. KU Leuven
"""Deals with Prolog-like programs."""

from .program import Program
from .program import RuleType
from .program import create_fact
from .program import create_query
from .program import create_rule
from .program import is_categorical_label
from .program import str_to_rule
from .program import str_to_rules
from .program import unwrap_categorical_label
from .transformation import expand_annotated_disjunctions
from .transformation import remove_labeled_rules


__all__ = [
    # Core types
    "Program",
    "RuleType",
    # Parsing (primary entry point for users)
    "str_to_rule",
    "str_to_rules",
    # Programmatic construction
    "create_rule",
    "create_query",
    "create_fact",
    # Transformations
    "expand_annotated_disjunctions",
    "remove_labeled_rules",
    # Categorical / annotated-disjunction label inspection
    "is_categorical_label",
    "unwrap_categorical_label",
]
