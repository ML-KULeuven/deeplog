#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog.util import broadcast_tensors
from deeplog.util import fast_dev_run_enabled


def test_broadcast_tensors():
    a = torch.tensor([[1]])  # (1, 1)
    b = torch.tensor([[1, 2, 3]])  # (1, 3)
    c = torch.tensor([[[5, 6], [7, 8]]])  # (1, 2, 2)

    a_b, b_b, c_b = broadcast_tensors(a, b, c)

    # Each tensor is expanded along dim 1 to the cartesian product 1*3*2 = 6,
    # with trailing dims preserved.
    assert a_b.shape == (1, 6)
    assert b_b.shape == (1, 6)
    assert c_b.shape == (1, 6, 2)
    torch.testing.assert_close(a_b, torch.tensor([[1, 1, 1, 1, 1, 1]]))
    torch.testing.assert_close(b_b, torch.tensor([[1, 2, 3, 1, 2, 3]]))
    torch.testing.assert_close(
        c_b,
        torch.tensor([[[5, 6], [5, 6], [5, 6], [7, 8], [7, 8], [7, 8]]]),
    )


def test_fast_dev_run_enabled(monkeypatch):
    monkeypatch.delenv("DEEPLOG_FAST_DEV_RUN", raising=False)
    monkeypatch.delenv("DEELOG_FAST_DEV_RUN", raising=False)
    assert fast_dev_run_enabled() is False


def test_fast_dev_run_enabled_truthy(monkeypatch):
    monkeypatch.delenv("DEELOG_FAST_DEV_RUN", raising=False)
    monkeypatch.setenv("DEEPLOG_FAST_DEV_RUN", "1")
    assert fast_dev_run_enabled() is True


def test_fast_dev_run_reads_the_old_spelling_while_the_new_one_is_unset(monkeypatch):
    monkeypatch.delenv("DEEPLOG_FAST_DEV_RUN", raising=False)
    monkeypatch.setenv("DEELOG_FAST_DEV_RUN", "1")
    assert fast_dev_run_enabled() is True


def test_fast_dev_run_reads_the_new_spelling_over_the_old_one(monkeypatch):
    monkeypatch.setenv("DEEPLOG_FAST_DEV_RUN", "0")
    monkeypatch.setenv("DEELOG_FAST_DEV_RUN", "1")
    assert fast_dev_run_enabled() is False
