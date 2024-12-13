import os
import random
from datetime import datetime

import yaml

from src import models
from src.datasets.cifar10 import get_cifar_10
from src.training import get_default_trainer


def set_up(arg):
    dataset = get_cifar_10(use_test_set=arg.test)
    model = models.load(arg.model_name)

    if arg.learning_rate_search:
        arg.lr = 10 ** random.uniform(-2., 0.)

    trainer = get_default_trainer(model, arg.lr, arg.epochs, arg.augmentation)
    return dataset, model, trainer


def save_results(**results):
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M_I.yaml")
    fp = os.path.join("outputs", "runs", filename)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    while os.path.exists(fp):
        fp = fp.replace(".yaml", "I.yaml")

    with open(fp, "w") as f:
        yaml.dump(results, f)
    print(f"Results: {results}")
