#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog.util import broadcast_tensors
from deeplog.util import fast_dev_run_enabled


def test_broadcast_tensors():
    a = torch.tensor([[1]])
    b = torch.tensor([[1, 2, 3]])
    c = torch.tensor([[[5, 6], [7, 8]]])
    broadcast_tensors(a, b, c)
    # TODO: check


def test_fast_dev_run_enabled(monkeypatch):
    monkeypatch.delenv("DEELOG_FAST_DEV_RUN", raising=False)
    assert fast_dev_run_enabled() is False


def test_fast_dev_run_enabled_truthy(monkeypatch):
    monkeypatch.setenv("DEELOG_FAST_DEV_RUN", "1")
    assert fast_dev_run_enabled() is True
