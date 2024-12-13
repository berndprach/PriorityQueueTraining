"""
Almost Orthogonal Lipschitz (AOL) layer.
Proposed in https://arxiv.org/abs/2208.03160
Code adapted from
"1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness", 2023.
"""
from functools import partial

import torch

from torch import Tensor, nn
from torch.nn.functional import conv2d, linear

from . import simple_conv, simple_linear
from .train_val_cache_decorator import train_val_cached
from .initializers import orthogonal_center


def rescale_kernel(weight: Tensor) -> Tensor:
    """ Expected weight shape: out_channels x in_channels x ks1 x ks_2 """
    _, _, k1, k2 = weight.shape
    weight_tp = weight.transpose(0, 1)
    v = torch.nn.functional.conv2d(
        weight_tp, weight_tp, padding=(k1 - 1, k2 - 1))
    v_scaled = v.abs().sum(dim=(1, 2, 3), keepdim=True).transpose(0, 1)
    return weight / (v_scaled + 1e-6).sqrt()


def rescale_matrix(weight: Tensor) -> Tensor:  # shape: out x in
    ls_bounds_squared = linear_bounds_squared(weight)
    return weight / (ls_bounds_squared + 1e-6).sqrt()  # shape: out x in


def linear_bounds_squared(weight: Tensor) -> Tensor:  # shape: out x in
    wwt = torch.matmul(weight.transpose(0, 1), weight)  # shape: in x in
    ls_bounds_squared = wwt.abs().sum(dim=0, keepdim=True)  # shape: 1 x in
    return ls_bounds_squared  # shape: out x in


class AOLConv2d(nn.Module):
    def __init__(self, *args, initializer=nn.init.dirac_, **kwargs):
        super().__init__()
        self.hp = simple_conv.Hp(*args, initializer=initializer, **kwargs)
        self.weight = simple_conv.get_kernel_parameter(self.hp)
        self.bias = simple_conv.get_bias_parameter(self.hp)

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.hp}."

    def forward(self, x: Tensor) -> Tensor:
        kernel = self.get_kernel()
        res = conv2d(x, kernel, bias=self.bias, padding=self.hp.padding)
        return res

    @train_val_cached
    def get_kernel(self):
        return rescale_kernel(self.weight)


AOLOrthogonalConv2d = partial(AOLConv2d, initializer=orthogonal_center)


class AOLLinear(nn.Module):
    def __init__(self, *args, initializer=nn.init.eye_, **kwargs):
        super().__init__()

        self.hp = simple_linear.Hp(*args, initializer=initializer, **kwargs)
        self.weight = simple_linear.get_weight_parameter(self.hp)
        self.bias = simple_linear.get_bias_parameter(self.hp)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.get_weight()
        return linear(x, weight, self.bias)

    @train_val_cached
    def get_weight(self):
        return rescale_matrix(self.weight)


AOLOrthogonalLinear = partial(AOLLinear, initializer=nn.init.orthogonal_)
