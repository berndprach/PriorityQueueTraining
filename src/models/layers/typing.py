from typing import Callable, Protocol

from torch import nn

LinearFactory = Callable[[int, int], nn.Module]
ChannelledFactory = Callable[[int], nn.Module]
PoolingFactory = ChannelledFactory
NormFactory = ChannelledFactory
ActivationFactory = Callable[[], nn.Module]


class ConvFactory(Protocol):
    def __call__(self,
                 c_in: int,
                 c_out: int,
                 kernel_size: tuple[int, int] = (3, 3),
                 ) -> nn.Module:
        ...
