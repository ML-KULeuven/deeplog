#  Copyright (c) 2024-2026. KU Leuven
# import torch
#
# from deeplog import NetworkPredicate
# from deeplog import SymTensor
# from ...testing_modules import IndexClassifier
#
#
# def test_network_predicate():
#     i1 = ("I1",)
#     i2 = ("I2",)
#     output_domain = [("a",), ("b",), ("c",)]
#
#     predicate = NetworkPredicate(
#         IndexClassifier(num_classes=len(output_domain)), "digit", 1, [(i1,), (i2,)]
#     )
#
#     expected_symbols = [("_", ("digit", i), ("probability",)) for i in (i1, i2)]
#     assert predicate.get_output_shape() == SymTensor(expected_symbols)
#
#     assignments = torch.tensor([[0.0, 1.0]])
#     output = predicate(assignments)
#
#     assert output.shape == (assignments.shape[0], 2, len(output_domain))
#
#     expected_output = torch.tensor(
#         [
#             [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]],
#         ]
#     )
#     torch.testing.assert_close(output, expected_output)
# TODO: Rework
