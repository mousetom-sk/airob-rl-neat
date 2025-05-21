from typing import Tuple, List

import argparse
from pathlib import Path

import numpy as np


def read_results(test_path: Path) -> Tuple[List[float], List[float], List[float], List[float]]:
    returns = []
    dists = []
    returns_std = []
    dists_std = []

    with open(test_path) as res:
        for line in res:
            ep = [float(x) for x in line.split(", ")]
            returns.append(ep[0])
            dists.append(ep[2])
            returns_std.append(ep[1])
            dists_std.append(ep[3])

    return returns, dists, returns_std, dists_std


parser = argparse.ArgumentParser()
parser.add_argument("--labels", help="table labels", nargs='+', type=str, required=True)
parser.add_argument("--paths", help="paths to the result directories", nargs='+', type=str, required=True)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    data = []

    for dir in args["paths"]:
        test_path = Path(f"{dir}/test.log")

        returns, dists, returns_std, dists_std = read_results(test_path)
        
        r_all, d_all = np.array(returns), np.array(dists)

        data.append({"avg": {"return": r_all.mean(),
                             "dist": d_all.mean()},
                     "std": {"return": r_all.std(),
                             "dist": d_all.std()}})
    
    for l, d in zip(args["labels"], data):
        row = l
        row += f' {d["avg"]["return"]:.3f} +/- {d["std"]["return"]:.3f}'
        row += f' {d["avg"]["dist"]:.3f} +/- {d["std"]["dist"]:.3f}'

        print(row)
