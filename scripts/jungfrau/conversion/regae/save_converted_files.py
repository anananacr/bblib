#!/usr/bin/env python3.6

import h5py
import argparse
import numpy as np
import om.utils.crystfel_geometry as crystfel_geometry
import cbf
import os
import subprocess as sub
from PIL import Image


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Convert JUNGFRAU 1M H5 images collected at REGAE for rotational data step/fly scan and return images in rotation sequence according tro the file index."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )

    parser.add_argument(
        "-g", "--geom", type=str, action="store", help="crystfel geometry file"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="hdf5 output path"
    )
    args = parser.parse_args(raw_args)

    raw_folder = os.path.dirname(args.input)
    output_folder = args.output
    cmd = f"cp {raw_folder}/info.txt {output_folder}"
    sub.call(cmd, shell=True)

    f = h5py.File(f"{args.input}_master.h5", "r")
    size = len(f["data"])

    label = (args.input).split("/")[-1]

    for i in range(size):
        try:
            raw = np.array(f["data"][i])
            raw[np.where(raw <= 0)] = -1
        except OSError:
            print("skipped", i)
            continue
        corr_frame = np.zeros(
            (1024,1024), dtype=np.int32
        )
        corr_frame = raw
        corr_frame[np.where(corr_frame <= 0)] = -1

        # cbf.write(f'{args.output}/{label}_{i:06}.cbf', corr_frame)
        Image.fromarray(corr_frame).save(f"{args.output}/{label}_{i:06}.tif")

    f.close()


if __name__ == "__main__":
    main()
