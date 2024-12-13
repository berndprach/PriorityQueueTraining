import math
from typing import NamedTuple, Callable

import torch
from torch import nn, Tensor
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from src.datasets import augmentation
from src.metrics import OffsetCrossEntropy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer(NamedTuple):
    model: nn.Module
    loss_function: Callable[[Tensor, Tensor], Tensor]
    optimizer: Optimizer
    scheduler: LRScheduler
    augment: Callable[[Tensor], Tensor]
    device: str = DEVICE


def step(trainer, model, batch):
    x_batch, y_batch = batch
    x_batch, y_batch = x_batch.to(trainer.device), y_batch.to(trainer.device)
    x_batch = trainer.augment(x_batch)
    predictions = model(x_batch)
    losses = trainer.loss_function(predictions, y_batch)
    loss = losses.mean()
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    return losses, predictions


def get_default_trainer(model, max_lr, epochs, aug_name="94perc"):
    loss_function = OffsetCrossEntropy(math.sqrt(2), 0.25, reduction="none")
    optimizer = SGD(model.parameters(), lr=0., momentum=0.9, nesterov=True)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=epochs)
    augment = augmentation.get(aug_name)
    return Trainer(model, loss_function, optimizer, scheduler, augment)


def to_device(batch, device):
    return tuple([tensor.to(device) for tensor in batch])
