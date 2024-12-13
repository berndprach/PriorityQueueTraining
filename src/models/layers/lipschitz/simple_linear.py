import math
from dataclasses import dataclass

from functools import partial
from typing import Optional, Callable

import torch
from torch import nn

torch_default_initializer = partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


@dataclass
class SimpleLinearHp:
    in_features: int
    out_features: int
    initializer: Optional[Callable] = torch_default_initializer
    bias: bool = False

    def get_weight_shape(self):
        return self.out_features, self.in_features


Hp = SimpleLinearHp


class SimpleLinear(nn.Module):
    """ Some defaults changed, initializer argument added. """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.hp = SimpleLinearHp(*args, **kwargs)

        self.weight = get_weight_parameter(self.hp)
        self.bias = get_bias_parameter(self.hp)

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.hp}."

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


def get_weight_parameter(hp: SimpleLinearHp):
    weight = nn.Parameter(torch.empty(hp.get_weight_shape()))
    hp.initializer(weight)
    return weight


def get_bias_parameter(hp: SimpleLinearHp):
    initial_bias = torch.zeros(hp.out_features)
    return nn.Parameter(initial_bias, requires_grad=hp.bias)
