import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics as m
from src.datasets.dataset import Dataset
from src.metric_tracker import MetricTracker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = (
    m.Accuracy(),
    m.OffsetCrossEntropy(),
    m.CRA(36 / 255),
    m.CRA(72 / 255),
    m.CRA(108 / 255),
    m.CRA(1 / 4),
    m.CRA(1 / 2),
    m.CRA(1),
)


def evaluate(model, dataset: Dataset, batch_size, metrics=METRICS):
    train_loader = DataLoader(dataset.train, batch_size=batch_size)
    train_results = evaluate_on_loader(model, train_loader, metrics, "train_")

    eval_loader = DataLoader(dataset.evaluate, batch_size=batch_size)
    eval_results = evaluate_on_loader(model, eval_loader, metrics, "val_")

    return {**train_results, **eval_results}


def evaluate_on_loader(model, loader, metrics, prefix="val_"):
    track = MetricTracker(metrics)
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        predictions = model(x_batch)
        track(predictions, y_batch)
    results = {
        prefix + metric.name: result
        for metric, result in zip(metrics, track.get_averages())
    }
    print(yaml.dump(results))
    return results
