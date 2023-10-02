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
    open_cc_map_global_max,
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
    minimum_snr=5,
    min_pixel_count=1,
    max_pixel_count=10,
    local_bg_radius=3,
    min_res=0,
    max_res=700,
)

RealCenter = [1255, 1158]

def calculate_cc(center:tuple)-> Dict[str, int]:
    # Update center for pf8 with the last calculated center
    # print(pf8_info)
    pf8_info.modify_radius(center[0], center[1])
    pf8_info._bad_pixel_map = mask
    pf8_info.adc_threshold = 10
    pf8_info.minimum_snr = 7
    pf8_info.max_res = 100
    pf8_info.min_pixel_count = 2
    pf8 = PF8(pf8_info)

    # Update geom and recorrect polarization
    updated_geom = f"{args.geom[:-5]}_{label}_{frame_number}_fwhm_{center[0]}_{center[1]}.geom"
    cmd = f"cp {args.geom} {updated_geom}"
    sub.call(cmd, shell=True)
    update_corner_in_geom(
        updated_geom, center[0], center[1]
    )
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        updated_geom, return_dict=True
    )
    corrected_data, _ = correct_polarization(x_map, y_map, clen_v, data, mask=mask)

    # Find Bragg peaks list with pf8
    
    peak_list = pf8.get_peaks_pf8(data=corrected_data)
    indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
    )
    transformed_indices=(indices[0]-center[1],indices[1]-center[0])
    """
    distance_to_center=[]
    for i in range(len(transformed_indices[0])):
        distance_to_center.append(math.sqrt(transformed_indices[0][i]**2+transformed_indices[1][i]**2))

    inverted_distance=distance_to_center[::-1]
    cc=np.corrcoef(distance_to_center,inverted_distance)[0,1]
    """
    inverted_indices=(transformed_indices[0][::-1], transformed_indices[1][::-1])
    cc_x=np.corrcoef(transformed_indices[1],inverted_indices[1])[0,1]
    cc_y=np.corrcoef(transformed_indices[0],inverted_indices[0])[0,1]

    return {
        "xc": center[0],
        "yc": center[1],
        "cc_x": -1*cc_x,
        "cc_y": -1*cc_y
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
                    xds_mask_sym[np.where(data > 1e5)] = 0
                    mask_sym = xds_mask_sym
                elif get_format(mask_sym_file_name) == "h":
                    f = h5py.File(f"{mask_sym_file_name}", "r")
                    mask_sym = np.array(f["data/data"])
                    mask_sym = np.array(mask_sym, dtype=np.int32)
                    # Mask of defective pixels
                    mask_sym[np.where(data < 0)] = 0
                    # Mask hot pixels
                    mask_sym[np.where(data > 1e5)] = 0
                    f.close()

            ## Peakfinder8 detector information and bad_pixel_map

            pf8_info.pf8_detector_info = dict(
                asic_nx=mask.shape[1],
                asic_ny=mask.shape[0],
                nasics_x=1,
                nasics_y=1,
            )
            pf8_info._bad_pixel_map = mask
            pf8_info.modify_radius(int(mask.shape[1] / 2), int(mask.shape[0] / 2))
            pf8_info.minimum_snr = 7
            pf8_info.min_pixel_count = 2
            pf8 = PF8(pf8_info)

            peak_list = pf8.get_peaks_pf8(data=data)
            indices = (
                np.array(peak_list["ss"], dtype=int),
                np.array(peak_list["fs"], dtype=int),
            )
            if peak_list["num_peaks"] >= MinPeaks:
                ## Peak search including more peaks
                pf8_info.pf8_detector_info = dict(
                    asic_nx=mask_sym.shape[1],
                    asic_ny=mask_sym.shape[0],
                    nasics_x=1,
                    nasics_y=1,
                )
                pf8_info._bad_pixel_map = mask_sym
                pf8_info.modify_radius(
                    int(mask_sym.shape[1] / 2), int(mask_sym.shape[0] / 2)
                )
                pf8_info.minimum_snr = 5
                pf8_info.min_pixel_count = 1
                pf8 = PF8(pf8_info)

                peak_list = pf8.get_peaks_pf8(data=data)
                indices = (
                    np.array(peak_list["ss"], dtype=int),
                    np.array(peak_list["fs"], dtype=int),
                )

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

                # Brute force manner

                ## Grid search of sharpness of the azimutal average

                pixel_step = 1
                xx, yy = np.meshgrid(
                    np.arange(xc - 60, xc + 61, pixel_step, dtype=int),
                    np.arange(yc - 60, yc + 61, pixel_step, dtype=int),
                )

                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                pool = multiprocessing.Pool()
                with pool:
                    cc_summary = pool.map(calculate_cc, coordinates)

                ## Display plots
                xc, yc = open_cc_map_global_max(
                    cc_summary, output_folder, f"{label}_{i}", pixel_step
                )
                last_center = (xc, yc)

                second_center = (xc, yc)

                results = calculate_cc((xc, yc))

                # Second mask peaks
                # Update geom and recorrect polarization
                updated_geom = f"{args.geom[:-5]}_{label}_{i}_v2.geom"
                cmd = f"cp {args.geom} {updated_geom}"
                sub.call(cmd, shell=True)
                update_corner_in_geom(updated_geom, xc, yc)
                cmd = f"cp {updated_geom} {output_folder}/final_geom "
                sub.call(cmd, shell=True)
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
                ax2.set_title("Second approximation: FWHM minimization")
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
                    f.create_dataset("id", data=file_name)
                    f.create_dataset("first_center", data=first_center)
                    f.create_dataset("second_center", data=second_center)
                    f.create_dataset("cc_x_max", data=results["cc_x"])
                    f.create_dataset("cc_y_max", data=results["cc_y"])
            else:
                if args.output:
                    f = h5py.File(f"{output_folder}/h5_files/{label}_{i}.h5", "w")
                    f.create_dataset("hit", data=0)
                    f.close()
            now = datetime.now()
            print(f"Current end time = {now}")


def open_cc_map_global_max(
    lines: list, output_folder: str, label: str, pixel_step: int
):
    """
    Open CC grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for CC optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
    """
    n = int(math.sqrt(len(lines)))

    merged_dict = {}
    for dictionary in lines[:]:

        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Create a figure with three subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z1 = np.array(merged_dict["cc_x"], dtype=np.float64).reshape((n, n))
    z2 = np.array(merged_dict["cc_y"], dtype=np.float64).reshape((n, n))
    
    pos1 = ax1.imshow(z1, cmap="rainbow")
    step = 20
    n = z1.shape[0]
    ax1.set_xticks(np.arange(step, n, step, dtype=int))
    ax1.set_yticks(np.arange(step, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax1.set_xticklabels(np.arange(x[0], x[-1], step, dtype=int))
    ax1.set_yticklabels(np.arange(y[0], y[-1], step, dtype=int))

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("CC")
    index_x = np.unravel_index(np.argmax(z1, axis=None), z1.shape)
    
    pos2 = ax2.imshow(z2, cmap="rainbow")
    step = 20
    n = z2.shape[0]
    ax2.set_xticks(np.arange(step, n, step, dtype=int))
    ax2.set_yticks(np.arange(step, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax2.set_xticklabels(np.arange(x[0], x[-1], step, dtype=int))
    ax2.set_yticklabels(np.arange(y[0], y[-1], step, dtype=int))

    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("CC")
    index_y = np.unravel_index(np.argmax(z2, axis=None), z2.shape)
    
    x = np.array(merged_dict["xc"]).reshape((n, n))
    y = np.array(merged_dict["yc"]).reshape((n, n))

    print(x[index_y],y[index_y], x[index_x],y[index_x])
    xc = x[index_x]
    yc = y[index_y]
    """
    proj_x = np.sum(z, axis=0)
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)
    index_x = np.unravel_index(np.argmax(proj_x, axis=None), proj_x.shape)
    xc = x[index_x]

    ax3.scatter(x, proj_x, color="b")
    ax3.scatter(xc, proj_x[index_x], color="r", label=f"xc: {xc}")
    ax3.set_ylabel("Average FWHM")
    ax3.set_xlabel("xc [px]")
    ax3.set_title("FWHM projection in x")
    ax3.legend()

    proj_y = np.sum(z, axis=1)
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    index_y = np.unravel_index(np.argmin(proj_y, axis=None), proj_y.shape)
    yc = x[index_y]
    ax4.scatter(proj_y, x, color="b")
    ax4.scatter(proj_y[index_y], yc, color="r", label=f"yc: {yc}")
    ax4.set_xlabel("Average FWHM")
    ax4.set_ylabel("yc [px]")
    ax4.set_title("FWHM projection in y")
    ax4.legend()
    
    """
    fig.colorbar(pos1, ax=ax1, shrink=0.6)
    fig.colorbar(pos2, ax=ax2, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f"{output_folder}/plots/cc_map/{label}.png")
    plt.close()
    return xc, yc



if __name__ == "__main__":
    main()

