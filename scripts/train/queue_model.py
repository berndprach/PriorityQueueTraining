import argparse
import sys

import torch

from src import models, training
from src.datasets import augmentation
from src.evaluate import evaluate
from src.set_up import set_up, save_results
from src.training_queues import TrainingQueue, HighestLossQueue

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

arg_parser = argparse.ArgumentParser()
add_arg = arg_parser.add_argument
add_arg("model_name", choices=models.names)
add_arg("epochs", type=int, nargs="?", default=24)
add_arg("-l", "--lr", type=float, default=0.1)
add_arg("-a", "--alpha", type=float, default=0.)
add_arg("--test", action="store_true")
add_arg("-lrs", "--learning-rate-search", action="store_true")
add_arg("-aug", "--augmentation", choices=augmentation.names, default="None")


def main(*args):
    arg = arg_parser.parse_args(args)
    print(f"Arguments: {arg}")

    dataset, model, trainer = set_up(arg)
    training_queue = HighestLossQueue(dataset.train, arg.alpha)

    train_model(model, trainer, training_queue, arg.epochs)
    results = evaluate(model, dataset, BATCH_SIZE)
    save_results(**results, **vars(arg))


def train_model(model, trainer, training_queue: TrainingQueue, epochs):
    number_of_batches = len(training_queue) // BATCH_SIZE
    model.to(DEVICE)

    for epoch_nr in range(epochs):
        print(f"Epoch {epoch_nr + 1}/{epochs}")
        for _ in range(number_of_batches):
            batch = training_queue.get_batch(BATCH_SIZE)
            losses, predictions = training.step(trainer, model, batch)
            training_queue.push_batch(losses, epoch_nr)

        trainer.scheduler.step()


if __name__ == "__main__":
    main(*sys.argv[1:])
