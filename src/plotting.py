import math
from pathlib import Path
from typing import Any, Iterable

import yaml
from matplotlib import pyplot as plt


def savefig(fp, *args, bbox_inches="tight", **kwargs):
    plt.savefig(fp, *args, bbox_inches=bbox_inches, **kwargs)
    print(f"Saved figure to {fp}")


Run = dict[str, Any]


def load_runs() -> list[Run]:
    runs = []
    for file in Path("outputs", "runs").iterdir():
        run = yaml.safe_load(file.read_text())
        runs.append(run)
    return runs


def filter_runs(runs: list[Run], conditions: dict[str, Any]) -> list[Run]:
    filtered_runs = []
    for run in runs:
        # if all(run.[key] == value for key, value in conditions.items()):
        if all(run.get(key, -1) == value for key, value in conditions.items()):
            filtered_runs.append(run)
    print(f"Filtered down to {len(filtered_runs)} by {conditions}")
    return filtered_runs


def split_by(key: str, runs: list[Run]) -> dict[Any, list[Run]]:
    values = set(run.get(key, None) for run in runs)
    split_runs = {
        value: [run for run in runs if run.get(key, None) == value]
        for value in values
    }
    data = {k: len(v) for k, v in split_runs.items()}
    print(f"Split by {key}: {data}")
    return split_runs


def make_unique(runs: list[Run], keys: Iterable[str]) -> list[Run]:
    unique_runs = []
    seen = set()
    for run in runs:
        key = tuple(run.get(k, None) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        unique_runs.append(run)
    print(f"Unique runs: {len(unique_runs)}")
    return unique_runs


def get(key: str, run: Run) -> Any:
    if key.startswith("log10."):
        return math.log10(run[key[6:]])
    return run[key]


def plot_line(x_values, y_values, alpha=0.3, **kwargs):
    x_values, y_values = zip(*sorted(zip(x_values, y_values)))
    plt.plot(x_values, y_values, alpha=alpha, **kwargs)


def set_percentage_ticks(axis="y"):
    if axis == "y":
        axis = plt.gca().yaxis
    axis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
