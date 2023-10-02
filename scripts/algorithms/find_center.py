#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
from datetime import datetime
import os
sys.path.append("/home/rodria/software/vdsCsPadMaskMaker/new-versions/")
import geometry_funcs as gf
import argparse
import numpy as np
from utils import (
    get_format,
    mask_peaks,
    center_of_mass,
    azimuthal_average,
    gaussian,
    open_fwhm_map,
    open_fwhm_map_global_min,
    fit_fwhm,
    shift_image_by_n_pixels,
    get_center_theory,
    correct_polarization,
    update_corner_in_geom,
)
import pandas as pd
from models import PF8, PF8Info
from scipy.optimize import curve_fit
import multiprocessing
import math
import matplotlib.pyplot as plt
from scipy import signal
import subprocess as sub
import h5py

MinPeaks = 15
global pf8_info

pf8_info = PF8Info(
    max_num_peaks=10000,
    adc_threshold=5,
    minimum_snr=7,
    min_pixel_count=2,
    max_pixel_count=10,
    local_bg_radius=3,
    min_res=0,
    max_res=1200
)

def calculate_fwhm(data_and_coordinates: tuple) -> Dict[str, int]:
    corrected_data, mask, center_to_radial_average = data_and_coordinates
    x, y = azimuthal_average(
        corrected_data, center=center_to_radial_average, mask=mask
    )
    x_all = x.copy()
    y_all = y.copy()
    # Plot all radial average
    if plot_flag:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(x, y)
    
    ## Define background peak region
    x_min = 200
    x_max = 400
    x = x[x_min:x_max]
    y = y[x_min:x_max]
    ## Estimation of initial parameters
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
        residuals = y - gaussian(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ## Calculation of FWHM
        fwhm = popt[2] * math.sqrt(8 * np.log(2))
        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]
    except:
        r_squared = 1
        fwhm = 800
        fwhm_over_radius = 800

    ## Display plots
    if plot_flag:
        x_fit = x.copy()
        y_fit = gaussian(x_fit, *popt)
        
        plt.vlines([x[0], x[-1]], 0,round(popt[0])+2, "r")

        plt.plot(
            x_fit,
            y_fit,
            "r--",
            label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
        )
        plt.title("Azimuthal integration")
        plt.xlim(0, 1200)
        plt.ylim(0, round(popt[0])+2)
        plt.legend()
        plt.savefig(
            f"{args.output}/plots/gaussian_fit/{label}_{frame_number}_{center_to_radial_average[0]}_{center_to_radial_average[1]}.png"
        )
        # plt.show()
        plt.close()

    return {
        "xc": center_to_radial_average[0],
        "yc": center_to_radial_average[1],
        "fwhm": fwhm,
        "fwhm_over_radius": fwhm_over_radius,
        "r_squared": r_squared,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate center of diffraction patterns fro MHz beam sweeping serial crystallography."
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
        "-g",
        "--geom",
        type=str,
        action="store",
        help="CrystFEL geometry filename",
    )


    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    global args
    args = parser.parse_args()
    global plot_flag
    plot_flag = False

    global frame_number

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    if args.mask:
        mask_files = open(args.mask, "r")
        mask_paths = mask_files.readlines()
        mask_files.close()
    else:
        mask_paths = []


    file_format = get_format(args.input)
    
    ### Extract geometry file
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        args.geom, return_dict=True
    )
    preamb, dim_info = gf.read_geometry_file_preamble(args.geom)
    dist_m = preamb["coffset"]
    res = preamb["res"]
    clen = preamb["clen"]
    dist = 0.0
    #print('Det',det_dict)
    global DetectorCenter
    DetectorCenter=[-1*det_dict['0']['corner_x'],-1*det_dict['0']['corner_y']]
    #print(DetectorCenter)
    output_folder = args.output
    global label
    label = (
        output_folder.split("/")[-1]
        + "_"
        + ((args.input).split("/")[-1]).split(".")[-1][3:]
    )

    if clen is not None:
        if not gf.is_float_try(clen):
            check = H5_name + clen
            myCmd = os.popen("h5ls " + check).read()
            if "NOT" in myCmd:
                # print("Error: no clen from .h5 file")
                clen_v = 0.0
            else:
                f = h5py.File(H5_name, "r")
                clen_v = f[clen][()] * (1e-3)  # f[clen].value * (1e-3)
                f.close()
                pol_bool = True
                # print("Take into account polarisation")
        else:
            clen_v = float(clen)
            pol_bool = True
            # print("Take into account polarisation")

        if dist_m is not None:
            dist_m += clen_v
        else:
            # print("Error: no coffset in geometry file. It is considered as 0.")
            dist_m = 0.0
        # print("CLEN, COFSET", clen, dist_m)
        dist = dist_m * res

    if file_format == "lst":
        ref_image = []
        for i in range(0, len(paths[:])):
        #for i in range(0, 3):

            file_name = paths[i][:-1]
            if len(mask_paths) > 0:
                mask_file_name = mask_paths[0][:-1]
            else:
                mask_file_name = False

            frame_number = i
            print(file_name)
                        
            initial_center = DetectorCenter

            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
            elif get_format(file_name) == "h":
                f = h5py.File(f"{file_name}", "r")
                data = np.array(f["raw_data"])
                f.close()

            if not mask_file_name:
                mask = np.ones(data.shape)
            else:
                if get_format(mask_file_name) == "cbf":
                    xds_mask = np.array(fabio.open(f"{mask_file_name}").data)
                    # Mask of defective pixels
                    xds_mask[np.where(xds_mask <= 0)] = 0
                    xds_mask[np.where(xds_mask > 0)] = 1
                    # Mask hot pixels
                    xds_mask[np.where(data > 1e5)] = 0
                    mask = xds_mask
                elif get_format(mask_file_name) == "h":
                    f = h5py.File(f"{mask_file_name}", "r")
                    mask = np.array(f["data/data"])
                    mask = np.array(mask, dtype=np.int32)
                    # Mask of defective pixels
                    mask[np.where(data < 0)] = 0
                    # Mask hot pixels
                    mask[np.where(data > 1e5)] = 0
                    f.close()

            ## Polarization correction factor
            corrected_data, pol_array_first = correct_polarization(
                x_map, y_map, clen_v, data, mask=mask
            )
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            pos1 = ax1.imshow(data * mask, vmax=50, cmap="cividis")
            pos2 = ax2.imshow(corrected_data * mask, vmax=50, cmap="cividis")
            pos3 = ax3.imshow(pol_array_first, vmin=0.7, vmax=1, cmap="cividis")
            ax1.set_title("Original data")
            ax2.set_title("Polarization corrected data")
            ax3.set_title("Polarization array")
            fig.colorbar(pos1, shrink=0.6, ax=ax1)
            fig.colorbar(pos2, shrink=0.6, ax=ax2)
            fig.colorbar(pos3, shrink=0.6, ax=ax3)
            # plt.show()
            plt.savefig(f"{args.output}/plots/pol/{label}_{i}.png")
            plt.close()
            
            ## Peakfinder8 detector information and bad_pixel_map

            pf8_info.pf8_detector_info = dict(
                asic_nx=mask.shape[1],
                asic_ny=mask.shape[0],
                nasics_x=1,
                nasics_y=1,
            )
            pf8_info._bad_pixel_map = mask
            pf8_info.modify_radius(DetectorCenter[0], DetectorCenter[1])
            pf8 = PF8(pf8_info)
            peaks_list = pf8.get_peaks_pf8(data=corrected_data)

            if peaks_list["num_peaks"] >= MinPeaks:
                first_center=center_of_mass(corrected_data, mask)

                ## Peak search including more peaks
                pf8_info.pf8_detector_info = dict(
                    asic_nx=mask.shape[1],
                    asic_ny=mask.shape[0],
                    nasics_x=1,
                    nasics_y=1,
                )
                pf8_info._bad_pixel_map = mask
                pf8_info.modify_radius(
                    int(first_center[0]), int(first_center[1])
                )
                pf8_info.minimum_snr = 5
                pf8_info.min_pixel_count = 1
                pf8 = PF8(pf8_info)

                peaks_list = pf8.get_peaks_pf8(data=corrected_data)
                indices = (
                    np.array(peaks_list["ss"], dtype=int),
                    np.array(peaks_list["fs"], dtype=int),
                )

                # Mask Bragg  peaks
                only_peaks_mask = mask_peaks(mask, indices, bragg=0)
                pf8_mask = only_peaks_mask * mask
                
                # Brute force manner
                ## Grid search of sharpness of the azimutal average
                """
                pixel_step = 1
                xx, yy = np.meshgrid(
                    np.arange(DetectorCenter[0] - 10, DetectorCenter[0] + 11, pixel_step, dtype=int),
                    np.arange(DetectorCenter[1] - 10, DetectorCenter[1] + 11, pixel_step, dtype=int)
                )
                """
                
                pixel_step = 1
                xx, yy = np.meshgrid(
                    np.arange(first_center[0] +10 - 15, first_center[0] +10 + 16, pixel_step, dtype=int),
                    np.arange(first_center[1] - 30 - 15, first_center[1] -30 + 16, pixel_step, dtype=int)
                )

                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                coordinates_anchor_data=[(corrected_data, pf8_mask, shift) for shift in coordinates]

                pool = multiprocessing.Pool()
                with pool:
                    fwhm_summary = pool.map(calculate_fwhm, coordinates_anchor_data)
                
                ## Display plots
                xc, yc = open_fwhm_map_global_min(
                    fwhm_summary, output_folder, f"{label}_{i}", pixel_step
                )
                last_center = (xc, yc)

                fit = open_fwhm_map(
                    fwhm_summary, output_folder, f"fine_{label}_{i}", pixel_step
                )
                if not fit:
                    xc, yc = last_center
                else:
                    xc, yc = fit

                refined_center = (xc, yc)
                print("Second approximation", xc, yc)

                plot_flag = True
                results = calculate_fwhm((corrected_data, pf8_mask,(xc, yc)))
                plot_flag = False

                # Second mask peaks
                # Update geom and move to final geometry files folder
                updated_geom = f"{args.geom[:-5]}_{label}_{i}_v1.geom"
                cmd = f"cp {args.geom} {updated_geom}"
                sub.call(cmd, shell=True)
                update_corner_in_geom(updated_geom, xc, yc)
                cmd = f"cp {updated_geom} {output_folder}/final_geom "
                sub.call(cmd, shell=True)
                
                ## Display plots
                xr = DetectorCenter[0]
                yr = DetectorCenter[1]

                fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
                pos1 = ax1.imshow(corrected_data, vmax=50, cmap="cividis")
                ax1.scatter(xr, yr, color="lime", label=f"Initial center:({xr},{yr})")
                ax1.scatter(
                    xc, yc, color="red", label=f"Refined center:({xc},{yc})"
                )
                ax1.set_title("Center refinement: FWHM minimization")
                fig.colorbar(pos1, ax=ax1, shrink=0.6)
                ax1.set_xlim(400, 2000)
                ax1.set_ylim(400, 2000)
                ax1.legend()
                plt.savefig(f"{args.output}/plots/centered/{label}_{i}.png")
                plt.close()
                # plt.show()
                
                if args.output:
                    f = h5py.File(f"{output_folder}/h5_files/{label}_{i}.h5", "w")
                    f.create_dataset("hit", data=1)
                    f.create_dataset("id", data=file_name)
                    f.create_dataset("first_center", data=first_center)
                    f.create_dataset("refined_center", data=refined_center)
                    f.create_dataset("fwhm_min", data=results["fwhm"])
                    f.create_dataset("r_squared_fwhm_min", data=results["r_squared"])
                    f.close()
                
            now = datetime.now()
            print(f"Current end time = {now}")


if __name__ == "__main__":
    main()
