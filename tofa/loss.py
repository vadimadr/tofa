from abc import ABC
from functools import partial
from typing import Any, Mapping

from torch import nn


class LossModule(nn.Module, ABC):
    """Base class for loss interface"""

    def forward(self, output: Mapping, batch_data: Mapping):
        raise NotImplementedError


class LossFunction(LossModule):
    """Wraps torchvision criterion"""

    loss = None

    def __init__(self, output_key="predict", target_key="target", **loss_kwargs):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key

        if isinstance(self.loss, type):
            self.loss_fn = self.loss(**loss_kwargs)
        else:
            self.loss_fn = partial(self.loss, **loss_kwargs)

    def forward(self, output, batch_data):
        output = output[self.output_key]
        label = batch_data[self.target_key]
        return self.loss_fn(output, label)


class MultiLoss(LossModule):
    def __init__(self, losses=None, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.losses = nn.ModuleDict()
        self.weights = {}

        if losses is not None:
            for loss in losses:
                self.add_loss(loss)

    def forward(self, output, batch_data):
        all_losses = {}
        for loss in self.losses:
            loss_val = self.losses[loss](output, batch_data)
            if isinstance(loss_val, dict):
                all_losses.update(loss_val)
            else:
                all_losses[loss] = loss_val
        total_loss = sum(self.weights[loss] * all_losses[loss] for loss in all_losses)
        if self.normalize:
            total_loss /= sum(self.weights.values())

        batch_data["losses"] = all_losses
        return total_loss

    def add_loss(self, loss, name=None, weight=1.0):
        if name is None:
            name = f"loss.{len(self.losses)}"

        self.losses[name] = loss
        self.weights[name] = weight
