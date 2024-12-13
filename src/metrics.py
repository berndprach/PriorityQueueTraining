import math

import torch
from torch.nn.functional import one_hot


class Accuracy:
    name = "Accuracy"
    print_as_percentage = True

    def __call__(self, prediction_batch, label_batch):
        predictions = prediction_batch.argmax(dim=1)
        if len(label_batch.shape) == 1:
            return torch.eq(predictions, label_batch).float()
        else:  # one-hot or soft labels
            return torch.eq(predictions, label_batch.argmax(dim=1)).float()


class CRA:  # Certified Robust Accuracy
    def __init__(self, radius: float):
        self.margin = radius * math.sqrt(2)
        self.name = f"CRA{radius:.2f}"
        super().__init__()

    def __call__(self, scores, labels):
        labels_oh = one_hot(labels, scores.shape[-1])
        penalized_scores = scores - self.margin * labels_oh
        penalized_predictions = penalized_scores.argmax(dim=1)
        return torch.eq(penalized_predictions, labels).float()


class OffsetCrossEntropy:
    def __init__(self, offset=math.sqrt(2), temperature=0.25, **kwargs):
        super().__init__()
        self.offset = offset
        self.temperature = temperature
        self.name = f"OX({offset:.2g}, {temperature:.2g})"
        self.std_xent = torch.nn.CrossEntropyLoss(**kwargs)

    def __call__(self, score_batch, label_batch):
        label_batch = to_one_hot(label_batch, score_batch.shape[-1])
        offset_scores = score_batch - self.offset * label_batch
        offset_scores /= self.temperature
        return self.std_xent(offset_scores, label_batch) * self.temperature


def to_one_hot(label_batch, num_classes, dtype=torch.float32):
    label_batch = torch.nn.functional.one_hot(
        label_batch.to(torch.int64),
        num_classes=num_classes,
    )
    label_batch = label_batch.to(dtype)
    return label_batch
