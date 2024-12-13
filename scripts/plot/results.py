import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src import models, plotting
from src.plotting import load_runs, filter_runs, get, plot_line, split_by

plt.rcParams.update({'font.size': 14})

arg_parser = argparse.ArgumentParser()
add_arg = arg_parser.add_argument
add_arg("model_name", choices=models.names)
add_arg("-x", default="epochs")
add_arg("-r", "--radius", type=int, default=36)
add_arg("-p", "--partition", choices=["train", "val"], default="train")
add_arg("--x-scale", choices=["linear", "log"], default="log")


PATH = Path("outputs", "plots")
PATH.mkdir(exist_ok=True)


Y = {
    36: "train_CRA0.14",
    72: "train_CRA0.28",
    108: "train_CRA0.42",
    255: "train_CRA1.00",
}


def main(*args):
    arg = arg_parser.parse_args(args)
    print(f"Arguments: {arg}")

    all_runs = load_runs()
    all_runs = filter_runs(all_runs, {
        "model_name": arg.model_name,
        "learning_rate_search": False,
    })

    split_runs = split_by("alpha", all_runs)
    y_key = Y[arg.radius].replace("train_", f"{arg.partition}_")
    for alpha, runs in split_runs.items():
        x_values = [get(arg.x, run) for run in runs]
        y_values = [get(y_key, run) for run in runs]

        label = "standard training" if alpha is None else f"$\\alpha={alpha}$"
        plt.scatter(x_values, y_values, label=label)
        plot_line(x_values, y_values)

    plt.xscale(arg.x_scale)

    plt.title(f"Fitting the training data")
    plt.xlabel(arg.x)
    plt.ylabel(f"training cra $\\epsilon={arg.radius}/255$")

    plotting.set_percentage_ticks()

    x_keys = sorted(set(get(arg.x, run) for run in all_runs))
    x_keys = x_keys[::2]
    plt.xticks(x_keys, [f"{x}" for x in x_keys])

    plt.legend()
    plt.grid()

    fp = PATH / f"{arg.model_name}_{arg.partition}_cra{arg.radius}.pdf"
    plotting.savefig(fp)

    plt.show()


