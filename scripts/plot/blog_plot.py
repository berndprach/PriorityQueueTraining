import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src import models, plotting
from src.plotting import (
    load_runs, filter_runs, get, plot_line, split_by, make_unique
)

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
    all_runs = [r for r in all_runs if r.get("epochs", 0) >= 32]
    all_runs = [r for r in all_runs if r.get("alpha", None) in [None, 0.1]]
    all_runs = make_unique(all_runs, keys=["alpha", "epochs"])

    split_runs = split_by("alpha", all_runs)
    y_key = Y[arg.radius].replace("train_", f"{arg.partition}_")
    for alpha, runs in split_runs.items():
        x_values = [get(arg.x, run) for run in runs]
        y_values = [get(y_key, run) for run in runs]

        label = "standard" if alpha is None else f"priority queue"
        plt.scatter(x_values, y_values, label=label)
        plot_line(x_values, y_values)

    plt.xscale(arg.x_scale)

    plt.title(f"Fitting the Training Data")
    plt.xlabel(arg.x.title())
    plt.ylabel(f"Training CRA" if arg.partition == "train" else "Test CRA")

    plotting.set_percentage_ticks(axis="y")

    x_ticks = [32, 128, 512, 2048, 8192]
    plt.xticks(x_ticks, [f"{x}" for x in x_ticks])

    plt.legend()
    plt.grid()

    # fp = PATH / f"blog_{arg.model_name}_{arg.partition}_cra{arg.radius}.pdf"
    fp = PATH / f"priority_queue_results_{arg.partition}_cra{arg.radius}.png"
    plotting.savefig(fp)

    plt.tight_layout()
    plt.show()


