#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import get_format
import statistics
import h5py


def main():
    parser = argparse.ArgumentParser(description="Plot calculated center distribution.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    file_format = get_format(args.input)
    output_folder = args.output
    label = "center_distribution_" + output_folder.split("/")[-1]

    print(label)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.set_title("Calculated direct beam position")
    ax1.set_xlabel("Direct beam position in x (px)")
    ax1.set_ylabel("Direct beam position in y (px)")
    center_x = []
    center_y = []
    if file_format == "lst":
        for i in paths:
            try:
                f = h5py.File(f"{i[:-1]}", "r")
                center = np.array(f["second_center"])
                center_x.append(center[0])
                center_y.append(center[1])
                f.close()
            except KeyError:
                print(i[:-1])
            except:
                print("OS", i[:-1])

            ax1.scatter(center[0], center[1], c="b", s=20)
    median_x = statistics.median(center_x)
    median_y = statistics.median(center_y)
    mean_x = statistics.mean(center_x)
    mean_y = statistics.mean(center_y)
    ax1.scatter(
        median_x,
        median_y,
        c="r",
        marker="X",
        s=50,
        label=f"median :{median_x, median_y}",
    )
    ax1.scatter(
        mean_x, mean_y, c="g", marker="s", s=50, label=f"mean :{mean_x, mean_y}"
    )
    fig.legend()
    # plt.show()
    plt.savefig(f"{args.output}/plots/{label}.png")
    plt.close()


if __name__ == "__main__":
    main()
