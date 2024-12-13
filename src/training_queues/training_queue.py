from typing import Protocol

from torch import Tensor

BatchIndices = list[int]
Batch = tuple[Tensor, Tensor]


class TrainingQueue(Protocol):
    def __len__(self) -> int:
        pass

    def get_batch(self, batch_size: int) -> Batch:
        pass

    def push_batch(self, losses: Tensor, epoch_nr: int):
        pass
