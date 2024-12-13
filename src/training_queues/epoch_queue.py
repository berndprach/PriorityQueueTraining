import random

from src.datasets.dataset import get_batch


class EpochQueue:
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.data_order = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.data_order)
        self.last_returned = -1

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, batch_size):
        batch_indices = [self.get_next_index() for _ in range(batch_size)]
        batch = get_batch(self.dataset, batch_indices)
        return batch

    def push_batch(self, losses, epoch_nr: int):
        pass

    def get_next_index(self):
        self.last_returned += 1
        self.last_returned = self.last_returned % len(self.dataset)
        return self.data_order[self.last_returned]
