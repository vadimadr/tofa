import torch
import numpy as np


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


def as_numpy(tensor_like) -> np.ndarray:
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like.detach().cpu().numpy()
    return np.asarray(tensor_like)


def as_scalar(scalar_like):
    if isinstance(scalar_like, torch.Tensor):
        return scalar_like.item()
    return np.asarray(scalar_like).reshape(1)[0]
