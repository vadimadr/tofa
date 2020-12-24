import re
import warnings
from collections import OrderedDict
from logging import getLogger
from shutil import copyfile

import torch
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel

from tofa.config import mutate_config
from tofa.filesystem import as_path, existing_path
from tofa.misc import getattr_nested

log = getLogger(__name__)


def load_state_dict(module, state_dict, classifier_layer_name=None, strict=True):
    """
    Robust checkpoint loading routine.

    Args:
        module: Loaded model
        state_dict: dict containing parameters and buffers
        classifier_layer_name (str): name of the classifier layer of the model
        strict (bool): whether exception should be raised when dicts mismatch
    """
    # unwrap from distributed data parallel
    if isinstance(module, (nn.DataParallel, DistributedDataParallel)):
        module = module.module

    # unwrap module in state dict
    unwrapped_state = OrderedDict()
    strip_line = "module."
    for k, v in state_dict.items():
        if k.startswith(strip_line):
            k = k[len(strip_line) :]
        unwrapped_state[k] = v

    if classifier_layer_name is None:
        return module.load_state_dict(unwrapped_state, strict=strict)

    module_classes = getattr_nested(module, classifier_layer_name).out_features
    checkpoint_classes = unwrapped_state[
        "{}.weight".format(classifier_layer_name)
    ].size(0)

    if module_classes != checkpoint_classes:
        warnings.warn(
            f"Number of classes in model and checkpoint vary ({module_classes} vs "
            f"{checkpoint_classes}). Do not loading last FC weights"
        )
        del unwrapped_state["{}.weight".format(classifier_layer_name)]
        del unwrapped_state["{}.bias".format(classifier_layer_name)]

    # check state dict intersection
    model_parameters = {key for key in module.named_parameters()}
    state_dict_parametes = set(unwrapped_state.keys())
    parameters_without_clf = model_parameters - {
        f"{classifier_layer_name}.weight",
        f"{classifier_layer_name}.bias",
    }
    if parameters_without_clf != state_dict_parametes:
        warnings.warn(
            "State dict parameters and model parameters mismatch, but loading with "
            "strict=False "
        )
    return module.load_state_dict(unwrapped_state, strict=False)


class CheckpointManager:
    def __init__(
        self, config, model, optimizer=None, scheduler=None, higher_is_better=False
    ):
        self.higher_is_better = higher_is_better
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.config = config
        self.checkpoint_path = existing_path(config.checkpoint_path)
        self.best_checkpoint_path = as_path(config.checkpoint_path) / "best.pth"
        self.last_checkpoint_path = as_path(config.checkpoint_path) / "last.pth"

        self._current_epoch = 0
        self._best_value = 0
        self._saved_checkpoint = None

    def restore_checkpoint(self, path=None, pretrain=False):
        if path is None:
            path = self._find_latest(self.checkpoint_path)
        elif path.is_dir():
            path = self._find_latest(path)

        log.info(f"Restore checkpoint from: {path}")

        checkpoint = torch.load(path.as_posix(), map_location="cpu")
        load_state_dict(self.model, checkpoint["state_dict"], strict=not pretrain)

        if pretrain:
            return

        if "optimizer" in checkpoint and self.optimizer is not None:
            load_state_dict(self.optimizer, checkpoint["optimizer"])
        if "scheduler" in checkpoint and self.scheduler is not None:
            load_state_dict(self.scheduler, checkpoint["scheduler"])
        if "last_epoch" in checkpoint:
            with mutate_config(self.config):
                self.config.training.start_epoch = checkpoint["last_epoch"]

    def save_checkpoint(self, path):
        if self._saved_checkpoint is not None:
            copyfile(self._saved_checkpoint.as_posix(), path.as_posix())

        checkpoint = {"state_dict": self.model.state_dict()}
        if self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dcit()
        checkpoint["best_acc"] = self._best_value
        checkpoint["last_epoch"] = self._current_epoch
        torch.save(checkpoint, path.as_posix())

        self._saved_checkpoint = path
        return path

    def end_epoch(self, epoch_number, current_value):
        self._current_epoch = epoch_number
        if self.config.checkpoints.save_last:
            self.save_checkpoint(self.last_checkpoint_path)
        if current_value > self._best_value and self.config.checkpoints.save_best:
            self.save_checkpoint(self.best_checkpoint_path)
        if epoch_number % self.config.checkpoints.keep_every == 0:
            self.save_checkpoint(self.checkpoint_path / f"{self._current_epoch}.pth")
        if self._is_better(current_value, self._best_value):
            self._best_value = current_value

        self._saved_checkpoint = None

    def _is_better(self, value, value_than):
        return self.higher_is_better == (value > value_than)

    def _find_latest(self, checkpoint_path):
        latest_path = checkpoint_path / "last.pth"
        if latest_path.exists():
            return latest_path
        latest_found = -1
        for ckpt_path in checkpoint_path.iterdir():
            ckpt_name = ckpt_path.name

            match = re.match(r".*_(\d+)\..*", ckpt_name)
            if match and int(match.group(1)) > latest_found:
                latest_found = int(match.group(1))
                latest_path = ckpt_path
        return latest_path
