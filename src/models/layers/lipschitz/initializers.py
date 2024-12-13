import torch
from torch import nn


def orthogonal_center(tensor: torch.nn.Parameter) -> None:
    """ K[:, :, 1, 1] = orthogonal, other entries are zero. """
    ks1, ks2 = tensor.shape[2:]
    tensor.data.fill_(0)
    nn.init.orthogonal_(tensor[:, :, ks1//2, ks2//2])
    return None
