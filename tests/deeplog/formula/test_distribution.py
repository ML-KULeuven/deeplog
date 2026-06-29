#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the boolean-to-probability leaf mapping."""

from deeplog.formula.distribution import build_leaf_mapping


class TestBuildLeafMapping:
    def test_maps_labeled_atom_to_its_probability_label(self):
        labels = {
            ("digit", ("i1",), ("0",)): ("classifier", ("i1",), ("0",)),
            ("digit", ("i1",), ("1",)): ("classifier", ("i1",), ("1",)),
        }
        mapping = build_leaf_mapping(labels)
        assert mapping(("_", ("digit", ("i1",), ("0",)), ("boolean",))) == (
            "_",
            ("classifier", ("i1",), ("0",)),
            ("probability",),
        )
        assert mapping(("_", ("digit", ("i1",), ("1",)), ("boolean",))) == (
            "_",
            ("classifier", ("i1",), ("1",)),
            ("probability",),
        )

    def test_atoms_sharing_arguments_are_unambiguous(self):
        """a(x1) and b(x1) share arguments; the labels disambiguate them.

        A by-arguments heuristic cannot resolve this — both atoms and both
        labels collide on arguments alone — but the label map is exact.
        """
        labels = {
            ("a", ("x1",)): ("nn1", ("x1",)),
            ("b", ("x1",)): ("nn2", ("x1",)),
        }
        mapping = build_leaf_mapping(labels)
        assert mapping(("_", ("a", ("x1",)), ("boolean",))) == (
            "_",
            ("nn1", ("x1",)),
            ("probability",),
        )
        assert mapping(("_", ("b", ("x1",)), ("boolean",))) == (
            "_",
            ("nn2", ("x1",)),
            ("probability",),
        )

    def test_unlabeled_leaf_is_retagged_to_probability(self):
        mapping = build_leaf_mapping({("a", ("x1",)): ("nn1", ("x1",))})
        # An unlabeled boolean leaf keeps its atom, retagged as probability.
        assert mapping(("_", ("fact", ("y",)), ("boolean",))) == (
            "_",
            ("fact", ("y",)),
            ("probability",),
        )

    def test_empty_labels_retag_everything(self):
        mapping = build_leaf_mapping({})
        assert mapping(("_", ("a",), ("boolean",))) == ("_", ("a",), ("probability",))

    def test_unwrapped_symbol_passes_through(self):
        mapping = build_leaf_mapping({})
        unknown = ("unknown",)
        assert mapping(unknown) == unknown
