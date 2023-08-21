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
    fill_gaps
)
import pandas as pd
from models import PF8, PF8Info
from scipy.optimize import curve_fit
import multiprocessing
import math
import matplotlib.pyplot as plt
from scipy import signal
import h5py


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

    file_format = get_format(args.input)
    if args.center:
        table_real_center, loaded_table = get_center_theory(paths, args.center)
    # (table_real_center)
    if file_format == "lst":
        ref_image = []
        for i in range(0, len(paths[:])):
            file_name = paths[i][:-1]
            global data
            global mask
            global frame_number
            frame_number=i
            print(file_name)
            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
                mask_file_name = mask_paths[0][:-1]
                xds_mask = np.array(fabio.open(f"{mask_file_name}").data)
                # Mask of defective pixels
                xds_mask[np.where(xds_mask <= 0)] = 0
                xds_mask[np.where(xds_mask > 0)] = 1
                # Mask hot pixels
                xds_mask[np.where(data > 1e3)] = 0
                mask = xds_mask
                real_center = table_real_center[i]

            
            ## Find peaks with peakfinder8 and mask peaks
            pf8_info = PF8Info(
                max_num_peaks=10000,
                pf8_detector_info=dict(
                    asic_nx=mask.shape[1],
                    asic_ny=mask.shape[0],
                    nasics_x=1,
                    nasics_y=1,
                ),
                adc_threshold=10,
                minimum_snr=5,
                min_pixel_count=1,
                max_pixel_count=200,
                local_bg_radius=3,
                min_res=0,
                max_res=10000,
                _bad_pixel_map=mask,
            )

            pf8 = PF8(pf8_info)

            peak_list = pf8.get_peaks_pf8(data=data)
            indices = (
                np.array(peak_list["ss"], dtype=int),
                np.array(peak_list["fs"], dtype=int),
            )
            # Mask Bragg  peaks

            only_peaks_mask = mask_peaks(mask, indices, bragg=0)
            bad_pixels_and_peaks_mask = only_peaks_mask * mask
            global pf8_mask
            pf8_mask = bad_pixels_and_peaks_mask
            global unbragged_data
            unbragged_data = data * pf8_mask

            ## Approximate center of mass
            xc, yc = center_of_mass(unbragged_data)

            ## Center of mass again with the flipped image to account for eventual background asymmetry

            flipped_data = unbragged_data[::-1, ::-1]
            xc_flip, yc_flip = center_of_mass(flipped_data)

            h, w = data.shape
            shift_x = w / 2 - xc
            shift_y = h / 2 - yc
            shift_x_flip = w / 2 - xc_flip
            shift_y_flip = h / 2 - yc_flip

            diff_x = abs((abs(shift_x) - abs(shift_x_flip)) / 2)
            diff_y = abs((abs(shift_y) - abs(shift_y_flip)) / 2)

            if shift_x <= 0:
                shift_x -= diff_x
            else:
                shift_x += diff_x
            if shift_y <= 0:
                shift_y -= diff_y
            else:
                shift_y += diff_y
            ## First approximation of the direct beam
            xc = int(round(w / 2 - shift_x))
            yc = int(round(h / 2 - shift_y))

            first_xc = xc
            first_yc = yc

            global center_x
            center_x = xc
            global center_y
            center_y = yc
            print("First approximation", xc, yc)

            ## Display first approximation plots
            """
            xr = real_center[0]
            yr = real_center[1]
            pos = plt.imshow(unbragged_data, vmax=7, cmap="jet")
            plt.scatter(xr, yr, color="lime", label="xds")
            plt.scatter(xc, yc, color="r", label="center of mass")
            plt.title("First approximation: center of mass")
            plt.colorbar(pos, shrink=0.6)
            plt.legend()
            #plt.savefig(
            #    f"/home/rodria/Desktop/20230814/com/lyso_{i}.png"
            #)
            plt.show()
            plt.close()
            """
            last_iter=(xc,yc)

            delta_center=2
            data_to_fill=data
            count=0
            center_iter=[]
            filled_data_iter=[]
            ave_r_iter=[]
            ave_y_iter=[]

            while delta_center>0.001 and count<=30:
                center_iter.append(last_iter)
                filled_data=fill_gaps(data_to_fill, last_iter, pf8_mask)
                filled_data_iter.append(filled_data.astype(np.int32))
                xc_fill, yc_fill = center_of_mass(filled_data)
            
                """
                Display plots
                fig, ax=plt.subplots(1,1, figsize=(10,10))
                ax.imshow(filled_data, vmax=7, cmap="jet")
                ax.scatter(xr, yr, color="lime", label=f"xds: ({xr}, {yr})")
                ax.scatter(last_iter[0], last_iter[1], color="purple", label=f"last iter: ({last_iter[0]}, {last_iter[1]})")
                ax.scatter(xc_fill, yc_fill, color="r", label=f"iter {count}: ({xc_fill}, {yc_fill})")
                plt.legend()
                plt.show()
                """
            
                delta_center=math.sqrt((xc_fill-last_iter[0])**2+(yc_fill-last_iter[1])**2)
                last_iter=[xc_fill, yc_fill]
                count+=1
                
            print(f"Last center: ({xc_fill},{yc_fill})")
            f=h5py.File(f"{args.output}_{i}.h5", "w")
            f.create_dataset("center_iter", data=center_iter)
            f.create_dataset("filled_data_iter", data=filled_data_iter)
            f.close()

if __name__ == "__main__":
    main()
