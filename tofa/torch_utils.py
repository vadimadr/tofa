from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tofa.misc import AttribDict

IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)


def as_tensor(array_like, ref_tensor=None, cpu=False) -> torch.Tensor:
    if isinstance(array_like, (float, int)):
        array_like = torch.Tensor([array_like])
    elif not isinstance(array_like, torch.Tensor):
        array_like = torch.Tensor(array_like)
    if ref_tensor is not None:
        array_like = array_like.to(ref_tensor.device, ref_tensor.dtype)
    if cpu:
        array_like.detach().cpu()
    return array_like


def as_numpy(tensor_like, dtype=None) -> np.ndarray:
    if isinstance(tensor_like, torch.Tensor):
        np_array = tensor_like.detach().cpu().numpy()
    else:
        np_array = np.asarray(tensor_like)
    if dtype is not None:
        np_array = np_array.astype(dtype)
    return np_array


def as_scalar(scalar_like):
    if isinstance(scalar_like, torch.Tensor):
        return scalar_like.item()
    return np.asarray(scalar_like).reshape(1)[0]


def tensor_to_image(
    tensor,
    mean=IMAGENET_MEAN_RGB,
    std=IMAGENET_STD_RGB,
    scale=255,
    to_bgr=False,
):
    mean = as_tensor(mean, tensor).view(-1, 1, 1)
    std = as_tensor(std, tensor).view(-1, 1, 1)
    denorm_tensor = (tensor * std + mean) * scale
    denorm_tensor = denorm_tensor.permute(1, 2, 0)
    image = as_numpy(denorm_tensor).astype(np.uint8)
    if to_bgr:
        image = image[..., ::-1]
    return image.copy()


def get_submodule(model, module_name, default=None):
    def _get_submodule_recursive(model, module_name, _full_name=None):
        head, tail = module_name.split(".", 1)
        for name, module in model.named_children():
            if name == module_name:
                return module
            if name == head:
                return _get_submodule_recursive(module, tail)

    module = _get_submodule_recursive(model, module_name)
    if module is None and default is None:
        raise ValueError(f"Module {module_name} not found in model")
    elif module is None:
        module = default
    return module


def set_submodule(model, module_name, value):
    def _set_submodule_recursive(model, module_name, value):
        head, tail = module_name.split(".", 1)
        for name, module in model.named_children():
            if name == module_name:
                model._modules[name] = value
                return True
            if name == head:
                return _set_submodule_recursive(model, module_name, value)
        return False

    status = _set_submodule_recursive(model, module_name, value)
    if not status:
        raise ValueError(f"Module {module_name} not found in model")


def children_nested(model: nn.Module):
    result = []

    def _flat_model_recursive(module: nn.Module):
        children = list(module.children())
        if not children:
            result.append(module)
        for child_module in module.children():
            _flat_model_recursive(child_module)

    _flat_model_recursive(model)
    return result


class NoCollateWrapper:
    """Wrapper for types that must be passed as is from dataloader (e.g. images)"""

    def __init__(self, obj):
        self.obj = obj


def collate(batch):
    """Collate function aware of certain custom types (e.g. Paths and SafeDicts)"""
    from torch.utils.data._utils.collate import default_collate

    elem = batch[0]
    elem_type = type(elem)
    # custom types
    if isinstance(elem, AttribDict):
        return AttribDict({key: collate([d[key] for d in batch]) for key in elem})
    if isinstance(elem, np.ndarray) and elem.dtype == np.object:
        return batch
    elif elem is None:
        return batch
    elif isinstance(elem, Path):
        return batch
    elif isinstance(elem, NoCollateWrapper):
        return [e.obj for e in batch]
    # default containers: correctly handle nested collation
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    # other types: fallback to default collate
    else:
        return default_collate(batch)
