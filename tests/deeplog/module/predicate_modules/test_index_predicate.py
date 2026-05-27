#  Copyright (c) 2024-2026. KU Leuven
# import torch
#
# from deeplog import IndexPredicate
# from deeplog import SymTensor
# from deeplog import reshape_input
#
#
# def test_index_predicate():
#     domain = [(str(i),) for i in range(10)]
#     arguments = [(("I1",), ("1",)), (("I2",), ("3",)), (("I1",), ("N",))]
#     predicate = IndexPredicate("nn", "probability", [domain], arguments)
#
#     expected_inputs = [
#         ("_", ("nn", ("I1",)), ("probability",)),
#         ("_", ("nn", ("I2",)), ("probability",)),
#     ]
#
#     assert set(predicate.get_input_shape()[0]) == set(expected_inputs)
#     assert predicate.get_input_shape()[1] == SymTensor([("N",)])
#
#     assert predicate.get_output_shape() == SymTensor(
#         [("_", ("nn", *args), ("probability",)) for args in arguments]
#     )
#
#     input_tensors = torch.tensor([[[0.2, 0.4, 0.1, 0.3], [0.1, 0.5, 0.2, 0.2]]])
#     expected = torch.tensor([[0.4, 0.2, 0.1]])
#
#     predicate = reshape_input(
#         predicate, (SymTensor(expected_inputs), SymTensor([("N",)]))
#     )
#     result = predicate.forward(input_tensors, torch.tensor([[2]]))
#     torch.testing.assert_close(result, expected)
# TODO: Rethink
