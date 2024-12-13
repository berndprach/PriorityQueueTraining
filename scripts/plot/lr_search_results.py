import argparse

import matplotlib.pyplot as plt

from src import models
from src.plotting import load_runs, filter_runs, get, plot_line

plt.rcParams.update({'font.size': 14})

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("model_name", choices=models.names)
arg_parser.add_argument("epochs", type=int, nargs="?", default=24)
arg_parser.add_argument("-a", "--alpha", type=float, default=0.)
arg_parser.add_argument("-x", default="log10.lr")
arg_parser.add_argument("-y", default="train_OX(1.4, 0.25)")


def main(*args):
    arg = arg_parser.parse_args(args)
    print(f"Arguments: {arg}")

    runs = load_runs()
    runs = filter_runs(runs, {
        "model_name": arg.model_name,
        "epochs": arg.epochs,
        "alpha": arg.alpha,
        "learning_rate_search": True,
    })

    x_values = [get(arg.x, run) for run in runs]
    y_values = [get(arg.y, run) for run in runs]

    print_minimum(x_values, y_values, arg.x)

    plt.scatter(x_values, y_values)
    plot_line(x_values, y_values)

    y_label = arg.y.replace("train_", "Train ")
    x_label = "Log10(LR)" if arg.x == "log10.lr" else arg.x.title()
    plt.title(f"{y_label} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


def print_minimum(x_values, y_values, x_key="x_key"):
    min_index = min(range(len(y_values)), key=y_values.__getitem__)
    min_x, min_y = x_values[min_index], y_values[min_index]
    print(f"Minimum for {x_key} = {min_x:.2f} (value {min_y:.2f})")
    if x_key.startswith("log10."):
        print(f"{x_key[6:]} = {10 ** min_x:.2f}")
