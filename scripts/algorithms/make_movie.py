#!/usr/bin/env python3.7
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as color
import imageio


def main():

    # argument parser
    parser = argparse.ArgumentParser(description="Gif maker.")
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="output folder images and gif"
    )
    args = parser.parse_args()
    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    images = []

    output_folder = args.output
    label = "center_calc_" + output_folder.split("/")[-1]
    for i in paths:
        images.append(imageio.imread(f"{i[:-1]}"))

    imageio.mimsave(f"{output_folder}/plots/{label}.gif", images, duration=1)


if __name__ == "__main__":
    main()
