import random
from _heapq import heapify, heappush, heappop
from typing import NamedTuple, Sequence, Any, Iterable

from torch import Tensor

from src.datasets.dataset import get_batch


Dataset = Sequence[tuple[Tensor, Any]]


class PrioritizedItem(NamedTuple):
    seen_before: int
    priority_number: float
    item: int


class HighestLossQueue:
    """
    Produces batches of indices in order of priority.
    ===
    First returns every element of the dataset once,
    then it returns the elements with the highest priority.
    """

    def __init__(self, dataset: Dataset, alpha=0., shuffle=True):
        self.dataset = dataset
        self.alpha = alpha

        indices = [i for i in range(len(dataset))]
        if shuffle:
            random.shuffle(indices)

        self.queue = [PrioritizedItem(0, p, i) for p, i in enumerate(indices)]
        heapify(self.queue)

        self.previous_indices = None

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, batch_size) -> tuple[Tensor, Tensor]:
        batch_indices = [self.pop() for _ in range(batch_size)]
        self.previous_indices = batch_indices
        batch = get_batch(self.dataset, batch_indices)
        return batch

    def push_batch(self, losses: Iterable[Tensor], epoch_nr):
        for index, loss in zip(self.previous_indices, losses):
            priority = loss.item() - self.alpha * epoch_nr
            self.push(index, priority)

    def push(self, index, priority):
        heappush(self.queue, PrioritizedItem(1, -priority, index))

    def pop(self):
        return heappop(self.queue).item
