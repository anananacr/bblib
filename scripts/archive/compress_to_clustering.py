#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
from datetime import datetime
import argparse
import numpy as np
from utils import get_format
from PIL import Image
import h5py


def main():
    parser = argparse.ArgumentParser(description="Compress hit diffraction patterns.")
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

    output_folder = args.output
    label = output_folder.split("/")[-1]
    file_format = get_format(args.input)
    if file_format == "lst":
        for i in range(0, len(paths[:])):
            file_name = paths[i][:-1]

            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
            elif get_format(file_name) == "h":
                f = h5py.File(f"{file_name}", "r")
                data = np.array(f["data"])
                f.close()

            im = Image.fromarray(data * 25.5).convert("L")
            im.save(f"{output_folder}/compressed/{label}_{i}.png", dpi=[300, 300])


if __name__ == "__main__":
    main()
