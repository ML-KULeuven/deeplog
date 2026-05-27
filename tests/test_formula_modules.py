#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog.module.aggregation_modules import AggregationModule
from deeplog.module.wrappers import WrappedModule
from deeplog.shape import SymTensor


def test_aggregation_module_creation():
    module = WrappedModule(lambda x: x, SymTensor(("X",)), SymTensor(("i", ("X",))))
    aggregation_module = AggregationModule(
        module, [("X",)], [torch.tensor([1, 2, 3])], "sum"
    )

    print(aggregation_module)
