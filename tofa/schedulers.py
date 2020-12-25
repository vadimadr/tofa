from bisect import bisect_right
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler:
    """Base class for schedulers"""

    def step(self, metric=None, step=None):
        raise NotImplemented


class ReduceLROnPlateau(_ReduceLROnPlateau, LRScheduler):
    def step(self, metric=None, step=None):
        super().step(metric)


def _check_milestones_list(milestones, schedulers):
    if not len(milestones) + 1 == len(schedulers):
        return False
    if not sorted(milestones) == milestones:
        return False
    if not len(set(milestones)) == len(milestones):
        return False
    return True


class ChainSchedulers(LRScheduler):
    def __init__(self, optimizer, milestones, schedulers):
        assert _check_milestones_list(milestones, schedulers)

        self.optimizer = optimizer
        self.milestones = milestones
        self.schedulers = schedulers
        self.last_epoch = -1

    def step(self, metric=None, step=None):
        if step is None:
            step = self.last_epoch + 1
        self.last_epoch = step

        scheduler = self.schedulers[bisect_right(self.milestones, step)]
        scheduler.step(metric, step)


class WarmUpWrapper(Optimizer):
    """Allows to use lr_scheduler before ending epoch"""

    def __init__(self, optimizer, scheduler, num_warmup_steps=None, last_step=-1):
        self.last_step = last_step
        self.num_warmup_steps = num_warmup_steps
        self.scheduler = scheduler
        self.optimizer = optimizer

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def add_param_group(self, param_group: dict) -> None:
        self.optimizer.add_param_group(param_group)

    def step(self, **kwargs):
        self.last_step = self.last_step + 1
        if self.last_step < self.num_warmup_steps:
            self.scheduler.step()
        self.optimizer.step(**kwargs)


class IterativeScheduler(_LRScheduler, LRScheduler):
    def step(self, metric=None, step=None):
        super().step()

    def get_lr(self):
        lrs = [self.compute_lr(self.last_epoch) for _ in self.optimizer.param_groups]
        return lrs

    def compute_lr(self, step):
        raise NotImplemented


class LinearLR(IterativeScheduler):
    def __init__(self, optimizer, start_lr, end_lr, num_steps):
        self.num_steps = num_steps
        self.end_lr = end_lr
        self.start_lr = start_lr
        super().__init__(optimizer)

    def compute_lr(self, step):
        r = step / self.num_steps
        lr = self.start_lr + r * (self.end_lr - self.start_lr)
        return max(lr, self.end_lr)


class ExponentialLR(IterativeScheduler):
    def __init__(self, optimizer, start_lr, end_lr, num_steps):
        self.num_steps = num_steps
        self.end_lr = end_lr
        self.start_lr = start_lr
        super().__init__(optimizer)

    def compute_lr(self, step):
        r = step / self.num_steps
        lr = (self.end_lr / self.start_lr) ** r
        return max(self.start_lr * lr, self.end_lr)


class StepLR(IterativeScheduler):
    def __init__(self, optimizer, step_size: Union[int, list], gamma, base_lr=None):
        if base_lr is None:
            base_lr = optimizer.param_groups[0]["lr"]

        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size
        self.constant_step = isinstance(step_size, int)

        if not self.constant_step:
            assert sorted(step_size) == step_size
            assert len(step_size) == len(set(step_size))

        super().__init__(optimizer)

    def compute_lr(self, step):
        if self.constant_step:
            r = step // self.step_size
        else:
            r = bisect_right(self.step_size, step)

        return self.base_lr * self.gamma ** r
