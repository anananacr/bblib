#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import numpy as np
from utils import (
    get_format,
    mask_peaks,
    center_of_mass,
    azimuthal_average,
    gaussian,
    open_fwhm_map,
    fit_fwhm,
    shift_image_by_n_pixels,
    get_center_theory,
    fill_gaps,
)
import pandas as pd
from models import PF8, PF8Info
from scipy.optimize import curve_fit
import multiprocessing
import math
import matplotlib.pyplot as plt
from scipy import signal
import h5py

global pf8_info

pf8_info = PF8Info(
    max_num_peaks=10000,
    adc_threshold=5,
    minimum_snr=6,
    min_pixel_count=2,
    max_pixel_count=200,
    local_bg_radius=3,
    min_res=0,
    max_res=10000,
)

global threshold_distance
# Distance in pixels of centers calculated between consecutive iterations.
threshold_distance = 0.5

global n_iterations
n_iterations = 50


def main():
    parser = argparse.ArgumentParser(
        description="Calculate center of diffraction patterns for MHz beam sweeping serial crystallography."
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
        "-m_sym",
        "--mask_sym",
        type=str,
        action="store",
        help="path to list of symmetric mask files .lst",
    )

    parser.add_argument(
        "-m_hr",
        "--mask_hr",
        type=str,
        action="store",
        help="path to list of symmetric mask files .lst",
    )
    parser.add_argument(
        "-center",
        "--center",
        type=str,
        action="store",
        help="path to list of theoretical center positions file in .txt",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    mask_files = open(args.mask, "r")
    mask_paths = mask_files.readlines()
    mask_files.close()

    mask_sym_files = open(args.mask_sym, "r")
    mask_sym_paths = mask_sym_files.readlines()
    mask_sym_files.close()

    mask_hr_files = open(args.mask_hr, "r")
    mask_hr_paths = mask_hr_files.readlines()
    mask_hr_files.close()

    file_format = get_format(args.input)
    if args.center:
        table_real_center, loaded_table = get_center_theory(paths, args.center)
    # (table_real_center)
    if file_format == "lst":
        ref_image = []
        for i in range(0, len(paths[:])):
            file_name = paths[i][:-1]

            print(file_name)
            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
                mask_file_name = mask_paths[i][:-1]
                mask_sym_file_name = mask_sym_paths[i][:-1]
                mask_hr_file_name = mask_hr_paths[i][:-1]
                xds_mask = np.array(fabio.open(f"{mask_file_name}").data)
                # Mask of defective pixels
                xds_mask[np.where(xds_mask <= 0)] = 0
                xds_mask[np.where(xds_mask > 0)] = 1
                # Mask hot pixels
                xds_mask[np.where(data > 1e3)] = 0
                mask = xds_mask

                # Mask of defective pixels

                xds_mask_sym = np.array(fabio.open(f"{mask_sym_file_name}").data)
                xds_mask_sym[np.where(xds_mask_sym <= 0)] = 0
                xds_mask_sym[np.where(xds_mask_sym > 0)] = 1
                # Mask hot pixels
                xds_mask_sym[np.where(data > 1e3)] = 0
                mask_sym = xds_mask_sym

                xds_mask_hr = np.array(fabio.open(f"{mask_hr_file_name}").data)
                xds_mask_hr[np.where(xds_mask_hr <= 0)] = 0
                xds_mask_hr[np.where(xds_mask_hr > 0)] = 1
                # Mask hot pixels
                # xds_mask_hr[np.where(data > 1e3)] = 0
                mask_hr = xds_mask_hr
                real_center = table_real_center[i]

            ## Peakfinder8 detector information and bad_pixel_map

            pf8_info.pf8_detector_info = dict(
                asic_nx=mask.shape[1],
                asic_ny=mask.shape[0],
                nasics_x=1,
                nasics_y=1,
            )
            pf8_info._bad_pixel_map = mask_sym
            pf8_info.modify_radius(int(mask.shape[1] / 2), int(mask.shape[0] / 2))
            pf8 = PF8(pf8_info)

            peak_list = pf8.get_peaks_pf8(data=data)
            indices = (
                np.array(peak_list["ss"], dtype=int),
                np.array(peak_list["fs"], dtype=int),
            )

            # Mask Bragg  peaks

            only_peaks_mask = mask_peaks(mask_sym, indices, bragg=0)
            pf8_mask = only_peaks_mask * mask_sym
            data_to_fill = data

            delta_center_x = 2
            delta_center_y = 2
            count = 0
            center_iter = []
            filled_data_iter = []
            last_iter = center_of_mass(data_to_fill * pf8_mask)
            center_iter.append(last_iter)
            filled_data_iter.append(data_to_fill * pf8_mask)

            # while delta_center_x>threshold_distance and delta_center_y>threshold_distance and count<n_iterations:
            while True and count < n_iterations:
                # Update center for pf8 with the last calculated center
                pf8_info.modify_radius(last_iter[0], last_iter[1])
                pf8_info._bad_pixel_map = mask

                # Find Bragg peaks list with pf8
                pf8 = PF8(pf8_info)
                peak_list = pf8.get_peaks_pf8(data=data)
                indices = (
                    np.array(peak_list["ss"], dtype=int),
                    np.array(peak_list["fs"], dtype=int),
                )

                # Mask Bragg  peaks
                only_peaks_mask = mask_peaks(mask, indices, bragg=0)
                pf8_mask = only_peaks_mask * mask

                # Fill gaps with the radial average with origin at the calculated center from last iteration
                filled_data = fill_gaps(data_to_fill, last_iter, pf8_mask)
                filled_data_iter.append(filled_data.astype(np.float32))
                xc_new, yc_new = center_of_mass(filled_data, mask_hr)

                center_iter.append([xc_new, yc_new])

                # Calculate distance in pixels from the last calculated center
                delta_center_x = abs(xc_new - last_iter[0])
                delta_center_y = abs(yc_new - last_iter[1])

                # Update center for next iteration
                last_iter = [xc_new, yc_new]
                count += 1

                """
                Display plots
                fig, ax=plt.subplots(1,1, figsize=(10,10))
                ax.imshow(filled_data, vmax=7, cmap="jet")
                ax.scatter(xr, yr, color="lime", label=f"xds: ({xr}, {yr})")
                ax.scatter(last_iter[0], last_iter[1], color="purple", label=f"last iter: ({last_iter[0]}, {last_iter[1]})")
                ax.scatter(xc_new, yc_new, color="r", label=f"iter {count}: ({xc_new}, {yc_new})")
                plt.legend()
                plt.show()
                """

            # print(f"Last center: ({xc_new},{yc_new})")
            f = h5py.File(f"{args.output}_{i}.h5", "w")
            f.create_dataset("center_iter", data=center_iter)
            f.create_dataset("filled_data_iter", data=filled_data_iter)
            f.close()


if __name__ == "__main__":
    main()
