import pytest
import torch
from torch import nn
from torch.optim import SGD

from tofa.schedulers import ExponentialLR, LinearLR, StepLR


def _make_optimizer(lr=0.1):
    params = [nn.Parameter(torch.zeros(1))]
    return SGD(params, lr=lr)


def _get_lr(optimizer: SGD):
    return optimizer.param_groups[0]["lr"]


def _step_n_epochs(scheduler, optimizer, n):
    lrs = []
    for i in range(n):
        lrs.append(_get_lr(optimizer))
        scheduler.step()
    return lrs


def test_linear_scheduler():
    optimizer = _make_optimizer()
    scheduler = LinearLR(optimizer, 0.1, 0.02, 4)

    lrs = _step_n_epochs(scheduler, optimizer, 6)
    assert pytest.approx([0.1, 0.08, 0.06, 0.04, 0.02, 0.02]) == lrs


def test_exp_scheduler():
    optimizer = _make_optimizer()
    scheduler = ExponentialLR(optimizer, 0.1, 0.00001, 4)

    lrs = _step_n_epochs(scheduler, optimizer, 6)
    assert pytest.approx([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-5]) == lrs


def test_fixed_step_scheduler():
    optimizer = _make_optimizer()
    scheduler = StepLR(optimizer, 2, 0.1)

    lrs = _step_n_epochs(scheduler, optimizer, 6)
    assert pytest.approx([1e-1, 1e-1, 1e-2, 1e-2, 1e-3, 1e-3]) == lrs


def test_multi_step_scheduler():
    optimizer = _make_optimizer()
    scheduler = StepLR(optimizer, [2, 4, 5], 0.1)

    lrs = _step_n_epochs(scheduler, optimizer, 7)
    assert pytest.approx([1e-1, 1e-1, 1e-2, 1e-2, 1e-3, 1e-4, 1e-4]) == lrs
