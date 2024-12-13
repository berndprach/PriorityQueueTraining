from dataclasses import dataclass
from functools import partial

from torch import nn

from . import layers
from .layers import typing as t


@dataclass
class LSCHyperparameters:
    base_width: int = 64
    nrof_classes: int = 10
    kernel_size: tuple[int, int] = (3, 3)
    nrof_blocks: int = 5
    block_depth: int = 3

    conv: t.ConvFactory = None
    linear: t.LinearFactory = None
    activation: t.ActivationFactory = layers.MaxMin


def get_lipschitz_conv_net(**kwargs) -> nn.Sequential:
    hp = LSCHyperparameters(**kwargs)

    def convolution(c_in: int, c_out: int, kernel_size=hp.kernel_size):
        return nn.Sequential(
            hp.conv(c_in, c_out, kernel_size=kernel_size),
            hp.activation(),
        )

    def down_sampling(c_in: int, c_out: int):
        return nn.Sequential(
            layers.FirstChannels(c_out // 4),
            nn.PixelUnshuffle(2),
        )

    backbone = nn.Sequential()
    c = hp.base_width
    for i in range(hp.nrof_blocks):
        for _ in range(hp.block_depth):
            # Note that ks=(3, 3) also in the last block,
            # so circular padding is illegal!
            backbone.append(convolution(c, c))
        backbone.append(down_sampling(c, 2 * c))
        c *= 2

    s = 32 // 2 ** hp.nrof_blocks

    return nn.Sequential(
        layers.ZeroChannelConcatenation(hp.base_width),
        convolution(hp.base_width, hp.base_width, kernel_size=(1, 1)),
        backbone,
        nn.MaxPool2d(s),
        nn.Flatten(),
        hp.linear(c, c),
        layers.FirstChannels(hp.nrof_classes),
    )


get_lsc = get_lipschitz_conv_net
get_aol_lsc = partial(get_lsc, conv=layers.AOLConv2d, linear=layers.AOLLinear)
