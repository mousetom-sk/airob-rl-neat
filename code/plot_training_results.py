from typing import Tuple, List

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 12, "figure.figsize": [9, 5]})


def read_results(log_path: str) -> Tuple[List[float], List[float]]:
    returns = []
    dists = []

    with open(log_path) as res:
        for line in res:
            ep = [float(x) for x in line.split(", ")]
            returns.append(ep[0])
            dists.append(ep[2])

    period = 10
    returns_moving = []
    dists_moving = []

    for i in range(len(returns)):
        returns_moving.append(np.array(returns[max(i -period + 1, 0):i + 1]).mean())
        dists_moving.append(np.array(dists[max(i - period + 1, 0):i + 1]).mean())

    return returns_moving, dists_moving


parser = argparse.ArgumentParser()
parser.add_argument("--labels", help="plot labels", nargs='+', type=str, required=True)
parser.add_argument("--paths", help="paths to the result directories", nargs='+', type=str, required=True)
parser.add_argument("--save-dir", help="path to the directory where to save plots", type=str, default=None)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    data = []

    for dir in args["paths"]:
        i = 0
        log_path = Path(f"{dir}/run_{i}.log")

        r_all, d_all = [], []

        while log_path.exists():
            returns, dists = read_results(log_path)
            r_all.append(returns)
            d_all.append(dists)

            i += 1
            log_path = Path(f"{dir}/run_{i}.log")
        
        r_all = np.array(r_all)
        d_all = np.array(d_all)

        data.append({"avg": {"returns": r_all.mean(0),
                             "dists": d_all.mean(0)},
                     "std": {"returns": r_all.std(0),
                             "dists": d_all.std(0)}})
    
    epochs = np.arange(max([len(d["avg"]["returns"]) for d in data]))

    for d in data:
        for stat in ("avg", "std"):
            for metric in ("returns", "dists"):
                d[stat][metric] = np.pad(d[stat][metric],
                                         (0, len(epochs) - len(d[stat][metric])),
                                         constant_values=np.nan)

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Return")

    for i, (l, d) in enumerate(zip(args["labels"], data)):
        ax.plot(epochs, d["avg"]["returns"], color=f"C{i}", label=l)
        
        if "std" in d:
            ax.fill_between(epochs,
                            d["avg"]["returns"] - d["std"]["returns"],
                            d["avg"]["returns"] + d["std"]["returns"],
                            color=f"C{i}", edgecolor=None, alpha=0.3)
        
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(0, len(epochs) + 1, 20))
    ax.set_xticks(np.arange(0, len(epochs) + 1, 10), minor=True)
    ax.set_xlim(0, len(epochs))
    ax.set_ylabel("Return")
    bottom = -50
    top = 100
    ax.set_yticks(np.arange(bottom, top + 1, 10))
    ax.set_yticks(np.arange(bottom, top + 1, 5), minor=True)
    ax.set_ylim(bottom, top)
    ax.legend(loc='best')
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(which="both")
    fig.tight_layout()
    
    if args["save_dir"]:
        fig.savefig(f'{args["save_dir"]}/return.pdf')
    else:
        plt.show()

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Distance")

    for i, (l, d) in enumerate(zip(args["labels"], data)):
        ax.plot(epochs, d["avg"]["dists"], color=f"C{i}", label=l)
        
        if "std" in d:
            ax.fill_between(epochs,
                            d["avg"]["dists"] - d["std"]["dists"],
                            d["avg"]["dists"] + d["std"]["dists"],
                            color=f"C{i}", edgecolor=None, alpha=0.3)
        
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(0, len(epochs) + 1, 20))
    ax.set_xticks(np.arange(0, len(epochs) + 1, 10), minor=True)
    ax.set_xlim(0, len(epochs))
    ax.set_ylabel("Distance")
    ax.set_yticks((np.arange(0, 21) - 10) / 10)
    ax.set_yticks((np.arange(0, 201, 5) - 100) / 100, minor=True)
    ax.set_ylim(-1, 1.01)
    ax.legend(loc='best')
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(which="both")
    fig.tight_layout()
    
    if args["save_dir"]:
        fig.savefig(f'{args["save_dir"]}/distance.pdf')
    else:
        plt.show()
