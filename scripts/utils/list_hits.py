#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
from datetime import datetime
import argparse
import numpy as np
from utils import get_format
import h5py


def main():
    parser = argparse.ArgumentParser(
        description="Make a list .lst of diffraction patterns classified as hits."
    )
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

    output_folder = args.output

    label = "split_hit_" + output_folder.split("/")[-1]
    print(label)

    f = h5py.File(f"{args.input}", "r")
    hit = np.array(f["hit"])
    raw_id = np.array(f["id"], dtype=str)
    f.close()

    raw_id = raw_id[np.where(hit == 1)]

    idx = 0

    for idx, i in enumerate(raw_id):
        if idx % 50 == 0:
            n = "{:02d}".format(1 + int(idx / 50))
            g = open(f"{args.output}/lists/{label}.lst{n}", "w")
        g.write(i)
    g.close()


if __name__ == "__main__":
    main()
