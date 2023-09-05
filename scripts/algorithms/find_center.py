#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
from datetime import datetime

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

MinPeaks = 25
global pf8_info

pf8_info = PF8Info(
    max_num_peaks=10000,
    adc_threshold=5,
    minimum_snr=5,
    min_pixel_count=1,
    max_pixel_count=10,
    local_bg_radius=3,
    min_res=0,
    max_res=700,
)

RealCenter = [1255, 1158]


def select_best_center(coordinates: list) -> list:
    fwhm_summary = []
    fwhm_over_radius_summary = []
    r_squared_summary = []
    good_coordinates = []
    directions_summary = []
    movement = ["+x", "-x", "+y", "-y", "0"]
    for idx, i in enumerate(coordinates):
        # Update center for pf8 with the last calculated center
        # print(pf8_info)
        pf8_info.modify_radius(i[0], i[1])
        pf8_info._bad_pixel_map = mask

        # Update geom and recorrect polarization
        updated_geom = (
            f"{args.geom[:-5]}_{label}_{frame_number}_fwhm_{i[0]}_{i[1]}.geom"
        )
        cmd = f"cp {args.geom} {updated_geom}"
        sub.call(cmd, shell=True)
        update_corner_in_geom(updated_geom, i[0], i[1])
        x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
            updated_geom, return_dict=True
        )
        corrected_data, _ = correct_polarization(x_map, y_map, clen_v, data, mask=mask)

        # Find Bragg peaks list with pf8
        pf8 = PF8(pf8_info)
        peak_list = pf8.get_peaks_pf8(data=corrected_data)
        indices = (
            np.array(peak_list["ss"], dtype=int),
            np.array(peak_list["fs"], dtype=int),
        )

        only_peaks_mask = mask_peaks(mask, indices, bragg=0)
        pf8_mask = only_peaks_mask * mask

        x, y = azimuthal_average(corrected_data, center=i, mask=pf8_mask)

        ## Define background peak region
        x_min = 100
        x_max = 450
        x = x[x_min:x_max]
        y = y[x_min:x_max]

        ## Estimation of initial parameters
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

        popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
        residuals = y - gaussian(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        ## Calculation of FWHM
        fwhm = popt[2] * math.sqrt(8 * np.log(2))

        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]
        # print(r_squared)
        if r_squared > 0.85:
            good_coordinates.append(i)
            fwhm_summary.append(fwhm)
            fwhm_over_radius_summary.append(fwhm_over_radius)
            r_squared_summary.append(r_squared)
            directions_summary.append(movement[idx])
    # print(good_coordinates)
    # print(np.array(fwhm_summary))
    sorted_candidates = sorted(
        list(
            zip(fwhm_summary, good_coordinates, r_squared_summary, directions_summary)
        ),
        key=lambda x: x[0],
    )
    fwhm_summary, good_coordinates, r_squared_summary, match_directions = zip(
        *sorted_candidates
    )
    if match_directions[0][-1] == "y" and match_directions[1][-1] == "x":
        xc = good_coordinates[1][0]
        yc = good_coordinates[0][1]
        results = calculate_fwhm((xc, yc))
        combined_fwhm = results["fwhm"]
        combined_fwhm_r_sq = results["r_squared"]
        if combined_fwhm < fwhm_summary[0]:
            best_center = [xc, yc]
            best_fwhm = combined_fwhm
            best_fwhm_r_squared = combined_fwhm_r_sq
        else:
            best_center = list(good_coordinates[0])
            best_fwhm = fwhm_summary[0]
            best_fwhm_r_squared = r_squared_summary[0]
    elif match_directions[0][-1] == "x" and match_directions[1][-1] == "y":
        xc = good_coordinates[0][0]
        yc = good_coordinates[1][1]
        combined_fwhm = calculate_fwhm((xc, yc))["fwhm"]
        if combined_fwhm < fwhm_summary[0]:
            best_center = [xc, yc]
            best_fwhm = combined_fwhm
            best_fwhm_r_squared = combined_fwhm_r_sq
        else:
            best_center = list(good_coordinates[0])
            best_fwhm = fwhm_summary[0]
            best_fwhm_r_squared = r_squared_summary[0]
    else:
        best_center = list(good_coordinates[0])
        best_fwhm = fwhm_summary[0]
        best_fwhm_r_squared = r_squared_summary[0]

    # print(best_center)
    return (best_center, best_fwhm, best_fwhm_r_squared)


def direct_search_fwhm(initial_center: list) -> Dict[str, int]:

    r = 0.9
    initial_step = 40
    step = initial_step
    last_center = initial_center
    next_center = initial_center
    center_pos_summary = [next_center]
    r_squared_summary = []
    fwhm_summary = []
    distance_x = 1
    distance_y = 1

    max_iter = 30
    n_iter = 0

    while step > 0 and n_iter < max_iter:
        # and distance_x > 0.5 and distance_y>0.5

        coordinates = [
            (next_center[0] + step, next_center[1]),
            (next_center[0] - step, next_center[1]),
            (next_center[0], next_center[1] + step),
            (next_center[0], next_center[1] - step),
            (next_center[0], next_center[1]),
        ]
        # print(coordinates)
        next_center, fwhm, r_squared = select_best_center(coordinates)
        center_pos_summary.append(next_center)
        fwhm_summary.append(fwhm)
        r_squared_summary.append(r_squared)
        step *= r
        step = int(step)
        distance_x, distance_y = (
            next_center[0] - last_center[0],
            next_center[1] - last_center[1],
        )
        last_center = next_center.copy()
        # print(distance_x,distance_y)
        n_iter += 1

    final_center = next_center

    return {
        "xc": final_center[0],
        "yc": final_center[1],
        "center_pos_summary": center_pos_summary,
        "fwhm_summary": fwhm_summary,
        "r_squared_summary": r_squared_summary,
    }


def calculate_fwhm(center_to_radial_average: tuple) -> Dict[str, int]:
    # Update center for pf8 with the last calculated center
    # print(pf8_info)
    pf8_info.modify_radius(center_to_radial_average[0], center_to_radial_average[1])
    pf8_info._bad_pixel_map = mask
    pf8_info.minimum_snr = 5
    pf8_info.min_pixel_count = 1

    # Update geom and recorrect polarization
    updated_geom = f"{args.geom[:-5]}_{label}_{frame_number}_fwhm_{center_to_radial_average[0]}_{center_to_radial_average[1]}.geom"
    cmd = f"cp {args.geom} {updated_geom}"
    sub.call(cmd, shell=True)
    update_corner_in_geom(
        updated_geom, center_to_radial_average[0], center_to_radial_average[1]
    )
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        updated_geom, return_dict=True
    )
    corrected_data, _ = correct_polarization(x_map, y_map, clen_v, data, mask=mask)

    # Find Bragg peaks list with pf8
    pf8 = PF8(pf8_info)
    peak_list = pf8.get_peaks_pf8(data=corrected_data)
    indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
    )

    only_peaks_mask = mask_peaks(mask, indices, bragg=0)
    pf8_mask = only_peaks_mask * mask

    x, y = azimuthal_average(
        corrected_data, center=center_to_radial_average, mask=pf8_mask
    )
    x_all = x.copy()
    y_all = y.copy()
    # Plot all radial average
    if plot_flag:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(x, y)
    r_squared = 1
    fwhm_r_sq_collection = []

    """
    ## Define background peak region
    ## Water ring
    x_min = 100
    x_max = 350
    x = x_all[x_min:x_max]
    y = y_all[x_min:x_max]

    ## Estimation of initial parameters
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
    except RuntimeError:
        fwhm = 800
        fwhm_over_radius = 800
        r_squared = 0
        popt=[max(y), mean, sigma]
    
    if r_squared>0:
        residuals = y - gaussian(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ## Calculation of FWHM
        fwhm = popt[2] * math.sqrt(8 * np.log(2))
        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]
    fwhm_r_sq_collection.append((fwhm, fwhm_over_radius, r_squared, popt))
    """
    ## Chip background
    x_min = 200
    x_max = 350
    x = x_all[x_min:x_max]
    y = y_all[x_min:x_max]
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
    except RuntimeError:
        fwhm = 800
        fwhm_over_radius = 800
        r_squared = 0
        popt = [max(y), mean, sigma]

    fwhm_r_sq_collection.append((fwhm, fwhm_over_radius, r_squared, popt))
    """
    ## Chip background
    x_min = 50
    x_max = 700
    x = x_all[x_min:x_max]
    y = y_all[x_min:x_max]
    ## Estimation of initial parameters
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
    except RuntimeError:
        fwhm = 800
        fwhm_over_radius = 800
        r_squared = 0
        popt=[max(y), mean, sigma]
    if r_squared>0:       
        residuals = y - gaussian(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ## Calculation of FWHM
        fwhm = popt[2] * math.sqrt(8 * np.log(2))
        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]
    fwhm_r_sq_collection.append((fwhm, fwhm_over_radius, r_squared, popt))
    
    ## Salt ring
    x_min = 200
    x_max = 400
    x = x_all[x_min:x_max]
    y = y_all[x_min:x_max]
    ## Estimation of initial parameters
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    try:   
        popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
    except RuntimeError:
        fwhm = 800
        fwhm_over_radius = 800
        r_squared = 0
        popt=[max(y), mean, sigma]
    if r_squared>0:       
        residuals = y - gaussian(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ## Calculation of FWHM
        fwhm = popt[2] * math.sqrt(8 * np.log(2))
        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]
    """
    fwhm_r_sq_collection.append((fwhm, fwhm_over_radius, r_squared, popt))
    fwhm_r_sq_collection.sort(key=lambda x: x[2], reverse=True)
    fwhm, fwhm_over_radius, r_squared, popt = fwhm_r_sq_collection[0]

    ## Display plots
    if plot_flag:
        x_fit = x.copy()
        y_fit = gaussian(x_fit, *popt)

        plt.plot(x, y)
        plt.plot(
            x_fit,
            y_fit,
            "r:",
            label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
        )
        plt.title("Azimuthal integration")
        plt.xlim(0, 1200)
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
        "pf8_mask": pf8_mask,
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
        "-center",
        "--center",
        type=str,
        action="store",
        help="path to list of theoretical center positions file in .txt",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    global args
    args = parser.parse_args()

    global data
    global mask
    global pf8_mask
    global frame_number
    global clen_v

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    if args.mask:
        mask_files = open(args.mask, "r")
        mask_paths = mask_files.readlines()
        mask_files.close()
    else:
        mask_paths = []

    if args.mask_sym:
        mask_sym_files = open(args.mask_sym, "r")
        mask_sym_paths = mask_sym_files.readlines()
        mask_sym_files.close()
    else:
        mask_sym_paths = []

    file_format = get_format(args.input)
    if args.center:
        table_real_center, loaded_table = get_center_theory(paths, args.center)
    else:
        table_real_center = []
    ### Extract geometry file
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        args.geom, return_dict=True
    )
    preamb, dim_info = gf.read_geometry_file_preamble(args.geom)
    dist_m = preamb["coffset"]
    res = preamb["res"]
    clen = preamb["clen"]
    dist = 0.0
    output_folder = args.output
    global label

    label = (
        output_folder.split("/")[-1]
        + "_"
        + ((args.input).split("/")[-1]).split(".")[-1][3:]
    )
    # print(label)
    global plot_flag
    plot_flag = False

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
            # for i in range(0, 1):

            file_name = paths[i][:-1]
            if len(mask_paths) > 0:
                mask_file_name = mask_paths[0][:-1]
            else:
                mask_file_name = False

            if len(mask_sym_paths) > 0:
                mask_sym_file_name = mask_sym_paths[0][:-1]
            else:
                mask_sym_file_name = False

            frame_number = i
            print(file_name)
            now = datetime.now()
            print(f"Current begin time = {now}")
            if len(table_real_center) > 0:
                real_center = table_real_center[i]
            else:
                real_center = RealCenter

            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
            elif get_format(file_name) == "h":
                f = h5py.File(f"{file_name}", "r")
                data = np.array(f["data"][4801])
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
                    xds_mask[np.where(data > 1e3)] = 0
                    mask = xds_mask
                elif get_format(mask_file_name) == "h":
                    f = h5py.File(f"{mask_file_name}", "r")
                    mask = np.array(f["data/data"])
                    mask = np.array(mask, dtype=np.int32)
                    # Mask of defective pixels
                    mask[np.where(data < 0)] = 0
                    # Mask hot pixels
                    mask[np.where(data > 1e3)] = 0
                    f.close()
            if not mask_sym_file_name and not mask_file_name:
                mask_sym = np.ones(data.shape)
            elif not mask_sym_file_name and mask_file_name:
                # mask_sym = (mask*mask[-1::-1]).copy()
                mask_sym = mask.copy()
            else:
                if get_format(mask_sym_file_name) == "cbf":
                    xds_mask_sym = np.array(fabio.open(f"{mask_sym_file_name}").data)
                    # Mask of defective pixels
                    xds_mask_sym[np.where(xds_mask_sym <= 0)] = 0
                    xds_mask_sym[np.where(xds_mask_sym > 0)] = 1
                    # Mask hot pixels
                    xds_mask_sym[np.where(data > 1e3)] = 0
                    mask_sym = xds_mask_sym
                elif get_format(mask_sym_file_name) == "h":
                    f = h5py.File(f"{mask_sym_file_name}", "r")
                    mask_sym = np.array(f["data/data"])
                    mask_sym = np.array(mask_sym, dtype=np.int32)
                    # Mask of defective pixels
                    mask_sym[np.where(data < 0)] = 0
                    # Mask hot pixels
                    mask_sym[np.where(data > 1e3)] = 0
                    f.close()

            ## Peakfinder8 detector information and bad_pixel_map

            pf8_info.pf8_detector_info = dict(
                asic_nx=mask.shape[1],
                asic_ny=mask.shape[0],
                nasics_x=1,
                nasics_y=1,
            )
            pf8_info._bad_pixel_map = mask_sym
            pf8_info.modify_radius(
                int(mask_sym.shape[1] / 2), int(mask_sym.shape[0] / 2)
            )
            pf8_info.minimum_snr = 7
            pf8_info.min_pixel_count = 2
            pf8 = PF8(pf8_info)

            peak_list = pf8.get_peaks_pf8(data=data)
            indices = (
                np.array(peak_list["ss"], dtype=int),
                np.array(peak_list["fs"], dtype=int),
            )
            if peak_list["num_peaks"] >= MinPeaks:
                # Mask Bragg  peaks
                only_peaks_mask = mask_peaks(mask_sym, indices, bragg=0)
                pf8_mask = only_peaks_mask * mask_sym
                ## Approximate center of mass
                xc, yc = center_of_mass(data, pf8_mask)
                first_masked_data = data.copy() * pf8_mask.copy()
                ## Update geometry file for nw x_map and y_map
                updated_geom = f"{args.geom[:-5]}_{label}_{i}_v1.geom"
                cmd = f"cp {args.geom} {updated_geom}"
                sub.call(cmd, shell=True)
                update_corner_in_geom(updated_geom, xc, yc)
                x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
                    updated_geom, return_dict=True
                )
                ## Center of mass again with the flipped image to account for eventual background asymmetry
                flipped_data = data[::-1, ::-1]
                flipped_mask = pf8_mask[::-1, ::-1]
                xc_flip, yc_flip = center_of_mass(flipped_data, flipped_mask)
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
                first_center = (xc, yc)
                print("First approximation", xc, yc)
                ## Display first approximation plots
                xr = real_center[0]
                yr = real_center[1]
                pos = plt.imshow(first_masked_data, vmax=10, cmap="jet")
                plt.scatter(xr, yr, color="lime", label=f"reference:({xr},{yr})")
                plt.scatter(
                    first_center[0],
                    first_center[1],
                    color="r",
                    label=f"calculated center:({first_center[0]}, {first_center[1]})",
                )
                plt.title("First approximation: center of mass")
                plt.colorbar(pos, shrink=0.6)
                plt.legend()
                plt.savefig(f"{args.output}/plots/com/{label}_{i}.png")
                # plt.show()
                plt.close()
                ## Correct for polarisation
                corrected_data, pol_array_first = correct_polarization(
                    x_map, y_map, clen_v, data, mask=mask
                )
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                pos1 = ax1.imshow(data * mask, vmax=10, cmap="jet")
                pos2 = ax2.imshow(corrected_data * mask, vmax=10, cmap="jet")
                pos3 = ax3.imshow(pol_array_first, vmin=0.7, vmax=1, cmap="jet")
                ax1.set_title("Original data")
                ax2.set_title("Polarization corrected data")
                ax3.set_title("Polarization array")
                fig.colorbar(pos1, shrink=0.6, ax=ax1)
                fig.colorbar(pos2, shrink=0.6, ax=ax2)
                fig.colorbar(pos3, shrink=0.6, ax=ax3)
                # plt.show()
                plt.savefig(f"{args.output}/plots/pol/{label}_{i}.png")
                plt.close()
                ## Second aproximation of the direct beam
                # Direct search method
                # results=direct_search_fwhm(list(first_center))
                # xc=results["xc"]
                # yc=results["yc"]
                # second_center = (xc, yc)
                # print("Second approximation", xc, yc)

                # Brute force manner
                ## Grid search of sharpness of the azimutal average
                pixel_step = 4
                xx, yy = np.meshgrid(
                    np.arange(xc - 52, xc + 53, pixel_step, dtype=int),
                    np.arange(yc - 52, yc + 53, pixel_step, dtype=int),
                )
                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                pool = multiprocessing.Pool()
                with pool:
                    fwhm_summary = pool.map(calculate_fwhm, coordinates)
                ## Display plots
                xc, yc = open_fwhm_map_global_min(
                    fwhm_summary, output_folder, f"{label}_{i}", pixel_step
                )
                last_center = (xc, yc)

                pixel_step = 1
                xx, yy = np.meshgrid(
                    np.arange(xc - 10, xc + 11, pixel_step, dtype=int),
                    np.arange(yc - 10, yc + 11, pixel_step, dtype=int),
                )
                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                pool = multiprocessing.Pool()
                with pool:
                    fwhm_summary = pool.map(calculate_fwhm, coordinates)
                ## Display plots
                fit = open_fwhm_map(
                    fwhm_summary, output_folder, f"fine_{label}_{i}", pixel_step
                )
                if not fit:
                    xc, yc = last_center
                else:
                    xc, yc = fit

                second_center = (xc, yc)
                print("Second approximation", xc, yc)

                plot_flag = True
                _ = calculate_fwhm((xc + 10, yc + 10))
                _ = calculate_fwhm((xc - 10, yc - 10))
                results = calculate_fwhm((xc, yc))

                plot_flag = False

                # Second mask peaks
                # Update geom and recorrect polarization
                updated_geom = f"{args.geom[:-5]}_{label}_{i}_v2.geom"
                cmd = f"cp {args.geom} {updated_geom}"
                sub.call(cmd, shell=True)
                update_corner_in_geom(updated_geom, xc, yc)
                x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
                    updated_geom, return_dict=True
                )
                corrected_data, pol_array_second = correct_polarization(
                    x_map, y_map, clen_v, data, mask=mask
                )
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                pos1 = ax1.imshow(data * mask, vmax=10, cmap="jet")
                ax1.set_title("Original data")
                pos2 = ax2.imshow(corrected_data * mask, vmax=10, cmap="jet")
                pos3 = ax3.imshow(pol_array_first, vmin=0.7, vmax=1, cmap="jet")
                ax2.set_title("Polarization corrected data")
                ax3.set_title("Polarization array")
                fig.colorbar(pos1, shrink=0.6, ax=ax1)
                fig.colorbar(pos2, shrink=0.6, ax=ax2)
                fig.colorbar(pos3, shrink=0.6, ax=ax3)
                # plt.show()
                plt.savefig(f"{args.output}/plots/pol_2/{label}_{i}.png")
                plt.close()
                pf8_info.pf8_detector_info = dict(
                    asic_nx=mask.shape[1],
                    asic_ny=mask.shape[0],
                    nasics_x=1,
                    nasics_y=1,
                )
                pf8_info._bad_pixel_map = mask
                pf8_info.modify_radius(xc, yc)
                pf8 = PF8(pf8_info)
                peak_list = pf8.get_peaks_pf8(data=corrected_data)
                indices = (
                    np.array(peak_list["ss"], dtype=int),
                    np.array(peak_list["fs"], dtype=int),
                )
                # Mask Bragg  peaks
                only_peaks_mask = mask_peaks(mask, indices, bragg=0)
                pf8_mask = only_peaks_mask * mask
                second_masked_data = data.copy() * pf8_mask.copy()
                ## Display plots
                xr = real_center[0]
                yr = real_center[1]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                pos1 = ax1.imshow(first_masked_data, vmax=10, cmap="jet")
                ax1.scatter(xr, yr, color="lime", label=f"reference:({xr},{yr})")
                ax1.scatter(
                    first_center[0],
                    first_center[1],
                    color="r",
                    label=f"calculated center:({first_center[0]},{first_center[1]})",
                )
                ax1.set_title("First approximation: center of mass")
                fig.colorbar(pos1, ax=ax1, shrink=0.6)
                ax1.legend()
                pos2 = ax2.imshow(second_masked_data, vmax=10, cmap="jet")
                ax2.scatter(xr, yr, color="lime", label=f"reference:({xr},{yr})")
                ax2.scatter(
                    xc, yc, color="blueviolet", label=f"calculated center:({xc},{yc})"
                )
                ax2.set_title("Second approximation: FWHM/R minimization")
                fig.colorbar(pos2, ax=ax2, shrink=0.6)
                ax2.legend()
                plt.savefig(f"{args.output}/plots/second/{label}_{i}.png")
                plt.close()
                # plt.show()

                ## Clean geom directory
                updated_geom = f"{args.geom[:-5]}_{label}_{i}_v2.geom"
                cmd = f"cp {updated_geom} {output_folder}"
                sub.call(cmd, shell=True)
                cmd = f"rm {output_folder}/../geom/{label[:-3]}/*{label}*.geom"
                sub.call(cmd, shell=True)
                cmd = (
                    f"mv {output_folder}/*v2.geom {output_folder}/../geom/{label[:-3]}"
                )
                sub.call(cmd, shell=True)

                if args.output:
                    f = h5py.File(f"{output_folder}/h5_files/{label}_{i}.h5", "w")
                    f.create_dataset("hit", data=1)
                    f.create_dataset("raw_data", data=data.astype(np.int32))
                    f.create_dataset("first_center", data=first_center)
                    f.create_dataset("second_center", data=second_center)
                    f.create_dataset("first_masked_data", data=first_masked_data)
                    f.create_dataset("second_masked_data", data=second_masked_data)
                    f.create_dataset("first_pol_array", data=pol_array_first)
                    f.create_dataset("second_pol_array", data=pol_array_second)
                    f.create_dataset("fwhm_min", data=results["fwhm"])
                    f.create_dataset("r_squared_fwhm_min", data=results["r_squared"])
                    f.close()
            else:
                if args.output:
                    f = h5py.File(f"{output_folder}/h5_files/{label}_{i}.h5", "w")
                    f.create_dataset("hit", data=0)
                    f.create_dataset("raw_data", data=data.astype(np.int32))
                    f.close()
            now = datetime.now()
            print(f"Current end time = {now}")


if __name__ == "__main__":
    main()
