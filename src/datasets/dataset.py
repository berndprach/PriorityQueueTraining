from typing import NamedTuple

import torch
from torchvision.datasets import VisionDataset


class Dataset(NamedTuple):
    train: VisionDataset
    evaluate: VisionDataset


def get_batch(training_dataset, batch_indices: list[int]):
    x_batch, y_batch = zip(*[training_dataset[i] for i in batch_indices])
    x_batch = torch.stack(x_batch)
    y_batch = torch.tensor(y_batch, dtype=torch.long)
    return x_batch, y_batch
