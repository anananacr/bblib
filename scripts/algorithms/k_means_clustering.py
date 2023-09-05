#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
from datetime import datetime
import argparse
import numpy as np
from utils import get_format
from PIL import Image
import matplotlib.pyplot as plt
import subprocess as sub
import multiprocessing
from sklearn.cluster import KMeans


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
        "-npi",
        "--numpy_input",
        type=str,
        action="store",
        help="path to list of X features extracted from images",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    data = np.load(f'{args.numpy_input}')
    reduced_data=data[:6000,:]
    norm_data=reduced_data.copy()
    row,col=reduced_data.shape
    for i in range(col):
        norm_factor=np.max(reduced_data[:,i])
        norm_data[:,i]=norm_data[:,i]//norm_factor
    kmeans = KMeans(n_clusters=2).fit(reduced_data)
    labels=kmeans.labels_

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(data)

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    for idx,i in enumerate(Z):
        cmd=f"cp {paths[idx][:-1]} {args.output}/cluster_{i}"
        #print(cmd)
        sub.call(cmd, shell=True)
    

if __name__ == "__main__":
    main()
