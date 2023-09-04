#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
from datetime import datetime
import argparse
import numpy as np
from utils import (
    get_format
)
from models import PF8, PF8Info 
import h5py
import multiprocessing

MinPeaks=15
global pf8_info

pf8_info = PF8Info(
    max_num_peaks=10000,
    adc_threshold=5,
    minimum_snr=7,
    min_pixel_count=2,
    max_pixel_count=10,
    local_bg_radius=3,
    min_res=0,
    max_res=1200,
)

def is_a_hit(file_name:str)->bool:
    file_name=file_name[:-1]
    if get_format(file_name) == "cbf":
        data = np.array(fabio.open(f"{file_name}").data)
    elif get_format(file_name) == "h":
        f = h5py.File(f"{file_name}", "r")
        data = np.array(f["data"][0])
        f.close()
    ## Mask defective pixels
    mask[np.where(data < 0)] = 0
    ## Mask hot pixels
    mask[np.where(data > 1e3)] = 0

    ## Peakfinder8 detector information and bad_pixel_map

    pf8_info.pf8_detector_info = dict(
        asic_nx=mask.shape[1],
        asic_ny=mask.shape[0],
        nasics_x=1,
        nasics_y=1,
        )
    pf8_info._bad_pixel_map = mask
    pf8_info.modify_radius(
        int(mask.shape[1] / 2), int(mask.shape[0] / 2)
    )
    pf8 = PF8(pf8_info)
    peak_list = pf8.get_peaks_pf8(data=data)
    indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
    )
    if peak_list["num_peaks"]>=MinPeaks:
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Calculate number of peaks in a diffraction patterns for hits pre filter and calculate the center of mass."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )

    parser.add_argument(
        "-m", "--mask", type=str, action="store", help="path to list of mask files .lst"
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()
    global mask

    if args.mask:
        mask_files = open(args.mask, "r")
        mask_paths = mask_files.readlines()
        mask_files.close()
    else:
        mask_paths = []

    file_format = get_format(args.input)
    output_folder = args.output
    label = "hit_"+output_folder.split("/")[-1]

    print(label)

    hit_log=[]

    if file_format == "lst":
        if len(mask_paths) > 0:
            mask_file_name = mask_paths[0][:-1]
        else:
            mask_file_name = False

        if not mask_file_name:
            print("Provide static mask file")
        else:
            if get_format(mask_file_name) == "cbf":
                xds_mask = np.array(fabio.open(f"{mask_file_name}").data)
                # Mask of defective pixels
                xds_mask[np.where(xds_mask <= 0)] = 0
                xds_mask[np.where(xds_mask > 0)] = 1
                mask = xds_mask

            elif get_format(mask_file_name) == "h":
                f = h5py.File(f"{mask_file_name}", "r")
                mask = np.array(f["data/data"])
                mask = np.array(mask, dtype=np.int32)
                f.close()

        for i in range(0, len(paths[:]),500):          
            file_name_chunk = paths[i:i+500]
            
            pool = multiprocessing.Pool()
            with pool:
                result_is_a_hit = pool.map(is_a_hit, file_name_chunk)
            
            hit_log+=result_is_a_hit

        hit_log=np.array(hit_log, dtype=int)

        if args.output:
            f = h5py.File(f"{output_folder}/{label}.h5", "w")
            f.create_dataset("hit", data=hit_log)
            f.create_dataset("id", data=paths)
            f.close()

if __name__== "__main__":
    main()