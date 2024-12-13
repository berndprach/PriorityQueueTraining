from dataclasses import dataclass
from functools import partial

from torch import nn

from src.models import layers
from src.models.layers import typing as t


@dataclass
class MLPHyperparameters:
    width: int = 32 * 32 * 3
    number_of_layers: int = 8
    number_of_classes: int = 10

    linear: t.LinearFactory = None
    activation: t.ActivationFactory = layers.MaxMin


def get_lipschitz_mlp(**kwargs):
    hp = MLPHyperparameters(**kwargs)

    model = nn.Sequential()
    model.append(nn.Flatten()),
    model.append(layers.ZeroChannelConcatenation(hp.width))

    for _ in range(hp.number_of_layers-1):
        model.append(hp.linear(hp.width, hp.width))
        model.append(hp.activation())
    model.append(hp.linear(hp.width, hp.width))

    model.append(layers.FirstChannels(10))
    return model


get_aol_mlp = partial(get_lipschitz_mlp, linear=layers.AOLLinear)

