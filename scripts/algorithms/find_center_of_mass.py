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

global pf8_info            

pf8_info = PF8Info(
    max_num_peaks=10000,
    adc_threshold=10,
    minimum_snr=7,
    min_pixel_count=2,
    max_pixel_count=10,
    local_bg_radius=3,
    min_res=0,
    max_res=10000
)

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

            
            ## Peakfinder8 detector information and bad_pixel_map

            pf8_info.pf8_detector_info=dict(
            asic_nx=mask.shape[1],
            asic_ny=mask.shape[0],
            nasics_x=1,
            nasics_y=1,
            )
            pf8_info._bad_pixel_map=mask
            
            delta_center=2
            count=0
            center_iter=[]
            filled_data_iter=[]
            last_iter=[]
            
            while delta_center>0.7 and count<=10:
                if not last_iter:
                    pf8_info.modify_radius(int(mask.shape[1]/2), int(mask.shape[0]/2))
                else:
                    pf8_info.modify_radius(last_iter[0], last_iter[1])
                pf8 = PF8(pf8_info)

                peak_list = pf8.get_peaks_pf8(data=data)
                indices = (
                np.array(peak_list["ss"], dtype=int),
                np.array(peak_list["fs"], dtype=int),
                )

                # Mask Bragg  peaks

                only_peaks_mask = mask_peaks(mask, indices, bragg=0)
                pf8_mask = only_peaks_mask * mask
                data_to_fill = data
                
                if last_iter:
                    filled_data=fill_gaps(data_to_fill, last_iter, pf8_mask)
                    filled_data_iter.append(filled_data.astype(np.float32))
                    xc_new, yc_new = center_of_mass(filled_data)
                    center_iter.append([xc_new,yc_new])
                    delta_center=math.sqrt((xc_new-last_iter[0])**2+(yc_new-last_iter[1])**2)
                    last_iter=[xc_new, yc_new]
                else:
                    last_iter = center_of_mass(data_to_fill*pf8_mask)
                    
                count+=1

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

            print(f"Last center: ({xc_new},{yc_new})")
            f=h5py.File(f"{args.output}_{i}.h5", "w")
            f.create_dataset("center_iter", data=center_iter)
            f.create_dataset("filled_data_iter", data=filled_data_iter)
            f.close()

if __name__ == "__main__":
    main()
