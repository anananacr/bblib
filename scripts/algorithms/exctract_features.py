#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
from datetime import datetime
import argparse
import numpy as np
from utils import get_format
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing


def exctract_features(file_name: str) -> list:
    im = Image.open(file_name[:-1])
    data = np.asarray(im)

    sum_total = np.sum(data)
    mask = np.where(data <= 0)
    nan_data = data.astype(float)
    nan_data[mask] = np.nan
    average_total = np.nanmean(nan_data)
    median_total = np.nanmedian(nan_data)
    std_total = np.nanstd(nan_data)
    var_total = np.nanvar(nan_data)

    cut_data = data.copy()
    mask = np.where(data > 100)
    cut_data[mask] = 0
    sum_bg_0 = np.sum(cut_data)
    nan_data = data.astype(float)
    nan_data[mask] = np.nan
    average_bg_0 = np.nanmean(nan_data)
    median_bg_0 = np.nanmedian(nan_data)
    std_bg_0 = np.nanstd(nan_data)
    var_bg_0 = np.nanvar(nan_data)

    cut_data = data.copy()
    mask = np.where(data > 200) and np.where(data < 100)
    cut_data[mask] = 0
    sum_bg_1 = np.sum(cut_data)
    average_bg_1 = np.nanmean(nan_data)
    median_bg_1 = np.nanmedian(nan_data)
    std_bg_1 = np.nanstd(nan_data)
    var_bg_1 = np.nanvar(nan_data)

    cut_data = data.copy()
    mask = np.where(data < 200)
    cut_data[mask] = 0
    sum_bg_2 = np.sum(cut_data)
    average_bg_2 = np.nanmean(nan_data)
    median_bg_2 = np.nanmedian(nan_data)
    std_bg_2 = np.nanstd(nan_data)
    var_bg_2 = np.nanvar(nan_data)

    return [
        sum_total,
        average_total,
        median_total,
        std_total,
        var_total,
        sum_bg_0,
        average_bg_0,
        median_bg_0,
        std_bg_0,
        var_bg_0,
        sum_bg_1,
        average_bg_1,
        median_bg_1,
        std_bg_1,
        var_bg_1,
        sum_bg_2,
        average_bg_2,
        median_bg_2,
        std_bg_2,
        var_bg_2,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Exctract features for clustering from compressed hit diffraction patterns."
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

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    output_folder = args.output
    label = output_folder.split("/")[-1]
    file_format = get_format(args.input)
    n=len(paths)

    if file_format == "lst":

        pool = multiprocessing.Pool()
        with pool:
            features_exctraction = pool.map(exctract_features, paths)
        features=np.array(features_exctraction).reshape((n,-1))

        if args.output:
            np.save(f"{output_folder}/{label}", features)
            
    

if __name__ == "__main__":
    main()
