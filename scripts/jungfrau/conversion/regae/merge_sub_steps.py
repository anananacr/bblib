import h5py
import argparse
import math
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, BinaryIO, List
import glob
import subprocess as sub
import os
import matplotlib.colors as color


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Convert JUNGFRAU 1M H5 images collected at REGAE for rotational data step/fly scan and return images in rotation sequence according tro the file index."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    parser.add_argument(
        "-n", "--n_frames", type=int, action="store", help="n frames to merge"
    )

    args = parser.parse_args(raw_args)

    fly_frames_to_merge = args.n_frames
    """Fly scan accumulate n sequential images"""
    f = h5py.File(f"{args.input}_master.h5", "r")
    size = len(f["/data"])
    n_frames_measured = math.floor(size / fly_frames_to_merge)
    averaged_frames = np.zeros((n_frames_measured, 1024, 1024), dtype=np.int32)
    acc_frame = np.zeros((1024, 1024), dtype=np.int32)
    count = 0
    for i in range(fly_frames_to_merge * n_frames_measured):
        raw = np.array(f["/data"][i])
        acc_frame += raw
        count += 1
        if (i + 1) % fly_frames_to_merge == 0:
            index = int((i + 1) / fly_frames_to_merge) - 1
            if count != 0:
                averaged_frames[index] = acc_frame / count
            acc_frame = np.zeros((1024, 1024), dtype=np.int32)
            count = 0
    f.close()

    g = h5py.File(f"{args.input}_merged_master.h5", "w")
    g.create_dataset("data", data=averaged_frames, compression="gzip")
    g.close()


if __name__ == "__main__":
    main()
