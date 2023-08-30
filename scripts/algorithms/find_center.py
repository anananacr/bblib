#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys

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

global pf8_info

pf8_info = PF8Info(
    max_num_peaks=10000,
    adc_threshold=5,
    minimum_snr=7,
    min_pixel_count=1,
    max_pixel_count=200,
    local_bg_radius=3,
    min_res=0,
    max_res=10000,
)


def calculate_fwhm(center_to_radial_average: tuple) -> Dict[str, int]:
    # Update center for pf8 with the last calculated center
    pf8_info.modify_radius(center_to_radial_average[0], center_to_radial_average[1])
    pf8_info._bad_pixel_map = mask

    # Update geom and recorrect polarization
    updated_geom = f"{args.geom[:-5]}_lyso_{frame_number}_fwhm_{center_to_radial_average[0]}_{center_to_radial_average[1]}.geom"
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
    # Plot all radial average
    # fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    # plt.plot(x, y)

    ## Define background peak region
    x_min = 150
    x_max = 400
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

    ## Display plots
    """
    x_fit=x.copy()
    y_fit=gaussian(x_fit, *popt)

    plt.plot(x,y)
    plt.plot(x_fit,y_fit, 'r:', label=f'gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}')
    plt.title('Azimuthal integration')
    plt.legend()
    plt.savefig(f'/home/rodria/Desktop/test/gaussian_fit/lyso_{frame_number}_{center_to_radial_average[0]}_{center_to_radial_average[1]}.png')
    #plt.show()
    plt.close()
    """

    return {
        "xc": center_to_radial_average[0],
        "yc": center_to_radial_average[1],
        "fwhm": fwhm,
        "fwhm_over_radius": fwhm_over_radius,
        "r_squared": r_squared,
        "pf8_mask": pf8_mask,
    }


def shift_and_calculate_cross_correlation(shift: Tuple[int]) -> Dict[str, float]:

    shift_x = -shift[0]
    shift_y = -shift[1]
    xc = round(data.shape[1] / 2) + shift[0]
    yc = round(data.shape[0] / 2) + shift[1]

    # Update geom and recorrect polarization
    updated_geom = f"{args.geom[:-5]}_lyso_{frame_number}_cc_{xc}_{yc}.geom"
    cmd = f"cp {args.geom} {updated_geom}"
    sub.call(cmd, shell=True)
    update_corner_in_geom(updated_geom, xc, yc)
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        updated_geom, return_dict=True
    )
    corrected_data, _ = correct_polarization(x_map, y_map, clen_v, data, mask=mask)

    shifted_data = shift_image_by_n_pixels(
        shift_image_by_n_pixels(corrected_data, shift_y, 0), shift_x, 1
    )
    shifted_mask = shift_image_by_n_pixels(
        shift_image_by_n_pixels(mask, shift_y, 0), shift_x, 1
    )

    # Update center for pf8 with the last calculated center
    pf8_info.modify_radius(
        round(corrected_data.shape[1] / 2), round(corrected_data.shape[0] / 2)
    )
    pf8_info._bad_pixel_map = shifted_mask
    pf8_info.minimum_snr = 8
    pf8_info.max_res = 200

    # Find Bragg peaks list with pf8
    pf8 = PF8(pf8_info)
    peak_list = pf8.get_peaks_pf8(data=shifted_data)

    flipped_data = shifted_data[::-1, ::-1]
    flipped_mask = shifted_mask[::-1, ::-1]
    pf8_info._bad_pixel_map = flipped_mask
    pf8 = PF8(pf8_info)

    peak_list_flipped = pf8.get_peaks_pf8(data=flipped_data)

    indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
    )

    indices_flipped = (
        np.array(peak_list_flipped["ss"], dtype=int),
        np.array(peak_list_flipped["fs"], dtype=int),
    )

    # Original image Bragg peak limits
    x_min_orig = np.min(indices[1])
    x_max_orig = np.max(indices[1])
    y_min_orig = np.min(indices[0])
    y_max_orig = np.max(indices[0])

    # Flipped image Bragg peak limits
    x_min_flip = np.min(indices_flipped[1])
    x_max_flip = np.max(indices_flipped[1])
    y_min_flip = np.min(indices_flipped[0])
    y_max_flip = np.max(indices_flipped[0])

    # Reduced dimensions
    x_min = min(x_min_orig, x_min_flip)
    x_max = max(x_max_orig, x_max_flip)
    y_min = min(y_min_orig, y_min_flip)
    y_max = max(y_max_orig, y_max_flip)

    # Construction of reduced original image
    img_1 = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
    x_orig = [x - x_min for x in indices[1]]
    y_orig = [x - y_min for x in indices[0]]

    orig_cut = (
        shifted_data[y_min : y_max + 1, x_min : x_max + 1]
        * shifted_mask[y_min : y_max + 1, x_min : x_max + 1]
    )
    ## Intensity weight
    img_1 = orig_cut * img_1

    ## Uncomment next line for no intensity weight
    # img_1[y_orig, x_orig]=1
    img_1 = mask_peaks(img_1, (y_orig, x_orig), 1)

    mask_1 = img_1.copy()
    mask_1[np.where(img_1 == 0)] = np.nan

    # Construction of reduced flipped image
    img_2 = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
    x_flip = [x - x_min for x in indices_flipped[1]]
    y_flip = [x - y_min for x in indices_flipped[0]]

    flip_cut = (
        flipped_data[y_min : y_max + 1, x_min : x_max + 1]
        * flipped_mask[y_min : y_max + 1, x_min : x_max + 1]
    )
    ## Intensity weight
    img_2 = flip_cut * img_2

    ## Uncomment next line for no intensity weight
    # img_2[y_flip, x_flip]=1
    img_2 = mask_peaks(img_2, (y_flip, x_flip), 1)

    mask_2 = img_2.copy()
    mask_2[np.where(img_2 == 0)] = np.nan

    # Calculate correlation matrix of the reduced image flipped in rlation to the original reduced image
    cc_matrix = correlate_2d(mask_1, mask_2)
    row, col = cc_matrix.shape
    row = round(row / 4)
    col = round(col / 4)
    reduced_cc_matrix = cc_matrix[row:-row, col:-col]

    # Reduction to the same shape as images in the central region of the cc matrix
    row, col = reduced_cc_matrix.shape
    row = round(row / 2)
    col = round(col / 2)

    ## Restriction due to the proximity of the center for spurious matches
    sub_reduced_cc_matrix = reduced_cc_matrix[row - 16 : row + 17, col - 16 : col + 17]

    ## Collect shifts from maximum and non zero values of the sub-reduced cc matrix
    if np.max(sub_reduced_cc_matrix) == 0:
        maximum_index = []
    else:
        maximum_index = np.where(sub_reduced_cc_matrix == np.max(sub_reduced_cc_matrix))
    non_zero_index = np.where(sub_reduced_cc_matrix != 0)
    index = np.unravel_index(
        np.argmax(np.abs(sub_reduced_cc_matrix)), sub_reduced_cc_matrix.shape
    )

    xx, yy = np.meshgrid(
        np.arange(-img_1.shape[1] / 2, img_1.shape[1] / 2, 1, dtype=np.int32),
        np.arange(-img_1.shape[0] / 2, img_1.shape[0] / 2, 1, dtype=np.int32),
    )
    xx = xx[row - 16 : row + 17, col - 16 : col + 17]
    yy = yy[row - 16 : row + 17, col - 16 : col + 17]

    ## Center track from corrections given by non-zero or maximum values in the sub reduced cc matrix
    orig_xc = xc
    orig_yc = yc

    max_candidates = []
    for index in zip(*maximum_index):
        max_candidates.append(
            [
                orig_xc + ((xx[index]) / 2),
                orig_yc + ((yy[index]) / 2),
                1 / math.sqrt(sub_reduced_cc_matrix[index]),
                index[0],
                index[1],
            ]
        )

    non_zero_candidates = []
    for index in zip(*non_zero_index):
        non_zero_candidates.append(
            [
                orig_xc + ((xx[index]) / 2),
                orig_yc + ((yy[index]) / 2),
                1 / math.sqrt(sub_reduced_cc_matrix[index]),
                index[0],
                index[1],
            ]
        )

    ## Sort candidates from distance to the initial center given and counts of overlaps in the cc matrix
    reference_point = (orig_xc, orig_yc)
    max_candidates.sort(
        key=lambda x: x[2]
        * math.sqrt((x[0] - reference_point[0]) ** 2 + (x[1] - reference_point[1]) ** 2)
    )
    non_zero_candidates.sort(
        key=lambda x: x[2]
        * math.sqrt((x[0] - reference_point[0]) ** 2 + (x[1] - reference_point[1]) ** 2)
    )

    ## Sort index list
    non_zero_index = []
    for candidate in non_zero_candidates:
        non_zero_index.append((candidate[3], candidate[4]))

    maximum_index = []
    for candidate in max_candidates:
        maximum_index.append((candidate[3], candidate[4]))

    ## Selection of best candidate for center approximation
    if len(non_zero_candidates) > 0:
        xc = non_zero_candidates[0][0]
        yc = non_zero_candidates[0][1]
        index = (non_zero_candidates[0][3], non_zero_candidates[0][4])
    elif len(max_candidates) > 0:
        xc = max_candidates[0][0]
        yc = maxx_candidates[0][1]
        index = (max_candidates[0][3], max_candidates[0][4])
    else:
        xc = 0
        yc = 0
        index = []

    ## Display plots

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    ax1.imshow(shifted_data * shifted_mask, vmax=10, cmap="cividis")
    ax1.scatter(indices[1], indices[0], facecolor="none", edgecolor="red")
    ax2.imshow(flipped_data * shifted_mask[::-1, ::-1], vmax=10, cmap="cividis")
    ax2.scatter(
        indices_flipped[1], indices_flipped[0], facecolor="none", edgecolor="lime"
    )
    ax3.imshow(orig_cut * img_1, cmap="jet", vmax=10)
    ax4.imshow(flip_cut * img_2, cmap="jet", vmax=10)
    plt.savefig(f"/home/rodria/Desktop/test/cc_flip/lyso_{frame_number}.png")
    # plt.show()
    plt.close()

    return {
        "max_index": np.array(maximum_index, dtype=np.int32),
        "non_zero_index": np.array(non_zero_index, dtype=np.int32),
        "index": np.array(index, dtype=np.int32),
        "xc": xc,
        "yc": yc,
        "sub_reduced_cc_matrix": np.array(sub_reduced_cc_matrix, dtype=np.int32),
        "xx": xx,
        "yy": yy,
        "max_candidates_com": max_candidates,
        "non_zero_candidates_com": non_zero_candidates,
    }


def correlate_2d(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Calculate cross correlation matrix between two images of same shape.
    The im2 will be slided over im1 and the product between overlap entries of both images corresponds to an element of the cross correlation matrix.

    Parameters
    ----------
    im1: np.ndarray
        Reference image, unvalid pixels must contain numpy.nan values.
    im2: np.ndarray
        Moving image, unvalid pixels must contain numpy.nan values.

    Returns
    ----------
    corr: np.ndarray
        Cross correlation matrix between im1 and im2.
    """

    corr = np.zeros((im1.shape[0] + im2.shape[0], (im1.shape[1] + im2.shape[1])))
    # Whole matrix
    # scan_boundaries_x = (-im1.shape[1],im1.shape[1])
    # scan_boundaries_y = (-im1.shape[0],im1.shape[0])

    # Fast mode
    scan_boundaries_x = (-16, 17)
    scan_boundaries_y = (-16, 17)

    xx, yy = np.meshgrid(
        np.arange(scan_boundaries_x[0], scan_boundaries_x[1], 1, dtype=int),
        np.arange(scan_boundaries_y[0], scan_boundaries_y[1], 1, dtype=int),
    )

    coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
    coordinates_anchor_data = [(coordinate, im1, im2) for coordinate in coordinates]
    pool = multiprocessing.Pool()
    with pool:
        cc_summary = pool.map(calculate_product, coordinates_anchor_data)

    corr_reduced = np.array(cc_summary).reshape((xx.shape))
    center_row = im1.shape[0]
    center_col = im1.shape[1]
    corr[
        center_row + scan_boundaries_y[0] : center_row + scan_boundaries_y[1],
        center_col + scan_boundaries_x[0] : center_col + scan_boundaries_x[1],
    ] = corr_reduced
    return corr


def calculate_product(key_args: tuple) -> float:
    """
    Calculate elements of the cross correlation matrix.
    The im2 will be slided over im1 by a shift of n pixels in both axis.
    The product is calculated from the not nan overlap entries of both images.

    Parameters
    ----------
    shift: Tuple[int]
        Coordinates of the sihft of the im2 in relation to im1.
    im1: np.ndarray
        Reference image for cross correlation
    im2: np.ndarray
        Moving image for cross correlation
    Returns
    ----------
    cc: float
        Element of the cross correlation matrix regarding the given shift.
    """
    shift, im1, im2 = key_args
    shift_x = shift[0]
    shift_y = shift[1]

    im2 = shift_image_by_n_pixels(shift_image_by_n_pixels(im2, shift_y, 0), shift_x, 1)
    im2[np.where(im2 == 0)] = np.nan
    cc = 0
    for idy, j in enumerate(im1):
        for idx, i in enumerate(j):
            if not np.isnan(i) and not np.isnan(im2[idy, idx]):
                cc += i * im2[idy, idx]
    return cc


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

    mask_sym_files = open(args.mask_sym, "r")
    mask_sym_paths = mask_sym_files.readlines()
    mask_sym_files.close()

    mask_files = open(args.mask, "r")
    mask_paths = mask_files.readlines()
    mask_files.close()

    file_format = get_format(args.input)
    if args.center:
        table_real_center, loaded_table = get_center_theory(paths, args.center)

    ### Extract geometry file
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        args.geom, return_dict=True
    )
    preamb, dim_info = gf.read_geometry_file_preamble(args.geom)
    dist_m = preamb["coffset"]
    res = preamb["res"]
    clen = preamb["clen"]
    dist = 0.0

    if clen is not None:
        if not gf.is_float_try(clen):
            check = H5_name + clen
            myCmd = os.popen("h5ls " + check).read()
            if "NOT" in myCmd:
                print("Error: no clen from .h5 file")
                clen_v = 0.0
            else:
                f = h5py.File(H5_name, "r")
                clen_v = f[clen][()] * (1e-3)  # f[clen].value * (1e-3)
                f.close()
                pol_bool = True
                print("Take into account polarisation")
        else:
            clen_v = float(clen)
            pol_bool = True
            print("Take into account polarisation")

        if dist_m is not None:
            dist_m += clen_v
        else:
            print("Error: no coffset in geometry file. It is considered as 0.")
            dist_m = 0.0
        print("CLEN, COFSET", clen, dist_m)

        dist = dist_m * res

    # (table_real_center)
    if file_format == "lst":
        ref_image = []
        for i in range(0, len(paths[:])):

            file_name = paths[i][:-1]

            frame_number = i
            print(file_name)
            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
                mask_file_name = mask_paths[i][:-1]
                mask_sym_file_name = mask_sym_paths[i][:-1]
                xds_mask = np.array(fabio.open(f"{mask_file_name}").data)

                # Mask of defective pixels
                xds_mask[np.where(xds_mask <= 0)] = 0
                xds_mask[np.where(xds_mask > 0)] = 1
                # Mask hot pixels
                xds_mask[np.where(data > 1e3)] = 0
                mask = xds_mask
                real_center = table_real_center[i]

                xds_mask_sym = np.array(fabio.open(f"{mask_sym_file_name}").data)
                xds_mask_sym[np.where(xds_mask_sym <= 0)] = 0
                xds_mask_sym[np.where(xds_mask_sym > 0)] = 1
                # Mask hot pixels
                xds_mask_sym[np.where(data > 1e3)] = 0
                mask_sym = xds_mask_sym

            elif get_format(file_name) == "h":
                f = h5py.File(f"{file_name}", "r")
                data = np.array(f["data"][4801])
                f.close()
                mask_file_name = mask_paths[0][:-1]
                f = h5py.File(f"{mask_file_name}", "r")
                mask = np.array(f["data/data"])
                mask = np.array(mask, dtype=np.int32)
                mask[np.where(data < 0)] = 0
                f.close()
                real_center = [537, 541]

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
            updated_geom = f"{args.geom[:-5]}_lyso_{i}_v1.geom"
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
            pos = plt.imshow(first_masked_data, vmax=7, cmap="jet")
            plt.scatter(xr, yr, color="lime", label=f"xds:({xr},{yr})")
            plt.scatter(
                first_center[0],
                first_center[1],
                color="r",
                label=f"calculated center:({first_center[0]}, {first_center[1]})",
            )
            plt.title("First approximation: center of mass")
            plt.colorbar(pos, shrink=0.6)
            plt.legend()
            plt.savefig(f"/home/rodria/Desktop/test/com/lyso_{i}.png")
            # plt.show()
            plt.close()

            ## Correct for polarisation

            corrected_data, pol_array_first = correct_polarization(
                x_map, y_map, clen_v, data, mask=mask
            )
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            pos1 = ax1.imshow(data * mask, vmax=7, cmap="jet")
            pos2 = ax2.imshow(corrected_data * mask, vmax=7, cmap="jet")
            pos3 = ax3.imshow(pol_array_first, vmin=0.7, vmax=1, cmap="jet")
            ax1.set_title("Original data")
            ax2.set_title("Polarization corrected data")
            ax3.set_title("Polarization array")
            fig.colorbar(pos1, shrink=0.6, ax=ax1)
            fig.colorbar(pos2, shrink=0.6, ax=ax2)
            fig.colorbar(pos3, shrink=0.6, ax=ax3)
            # plt.show()
            plt.savefig(f"/home/rodria/Desktop/test/pol/lyso_{i}.png")
            plt.close()

            ## Grid search of sharpness of the azimutal average

            xx, yy = np.meshgrid(
                np.arange(xc - 30, xc + 31, 1, dtype=int),
                np.arange(yc - 30, yc + 31, 1, dtype=int),
            )

            coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))

            pool = multiprocessing.Pool()
            with pool:
                fwhm_summary = pool.map(calculate_fwhm, coordinates)

            ## Display plots
            # open_fwhm_map(fwhm_summary, i)

            ## Second aproximation of the direct beam

            xc, yc = fit_fwhm(fwhm_summary)
            print("Second approximation", xc, yc)
            second_center = (xc, yc)
            # Second mask peaks
            # Update geom and recorrect polarization
            updated_geom = f"{args.geom[:-5]}_lyso_{i}_v2.geom"
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
            pos1 = ax1.imshow(data * mask, vmax=7, cmap="jet")
            ax1.set_title("Original data")
            pos2 = ax2.imshow(corrected_data * mask, vmax=7, cmap="jet")
            pos3 = ax3.imshow(pol_array_first, vmin=0.7, vmax=1, cmap="jet")
            ax2.set_title("Polarization corrected data")
            ax3.set_title("Polarization array")
            fig.colorbar(pos1, shrink=0.6, ax=ax1)
            fig.colorbar(pos2, shrink=0.6, ax=ax2)
            fig.colorbar(pos3, shrink=0.6, ax=ax3)
            # plt.show()
            plt.savefig(f"/home/rodria/Desktop/test/pol_2/lyso_{i}.png")
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
            pos1 = ax1.imshow(first_masked_data, vmax=7, cmap="jet")
            ax1.scatter(xr, yr, color="lime", label=f"xds:({xr},{yr})")
            ax1.scatter(
                first_center[0],
                first_center[1],
                color="r",
                label=f"calculated center:({first_center[0]},{first_center[1]})",
            )
            ax1.set_title("First approximation: center of mass")
            fig.colorbar(pos1, ax=ax1, shrink=0.6)
            ax1.legend()

            pos2 = ax2.imshow(second_masked_data, vmax=7, cmap="jet")
            ax2.scatter(xr, yr, color="lime", label=f"xds:({xr},{yr})")
            ax2.scatter(
                xc, yc, color="blueviolet", label=f"calculated center:({xc},{yc})"
            )
            ax2.set_title("Second approximation: FWHM/R minimization")
            fig.colorbar(pos2, ax=ax2, shrink=0.6)
            ax2.legend()
            plt.savefig(f"/home/rodria/Desktop/test/second/lyso_{i}.png")
            plt.close()
            # plt.show()

            ## Third approximation autocorrelation of Bragg peaks position
            # Calculate shift to closest center know so far

            shift = [int(-(data.shape[1] / 2) + xc), int(-(data.shape[0] / 2) + yc)]

            # shift = [
            #    int(-(data.shape[1] / 2) + real_center[0]),
            #    int(-(data.shape[0] / 2) + real_center[1]),
            # ]

            results = shift_and_calculate_cross_correlation(shift)
            third_center = (results["xc"], results["yc"])
            third_masked_data = np.array(corrected_data * mask, dtype=np.int32)
            f = h5py.File(f"{args.output}_{i}.h5", "w")

            if args.output:
                for key in results:
                    f.create_dataset(key, data=results[key])

                f.create_dataset("first_center", data=first_center)
                f.create_dataset("second_center", data=second_center)
                f.create_dataset("third_center", data=third_center)
                f.create_dataset("first_masked_data", data=first_masked_data)
                f.create_dataset("second_masked_data", data=second_masked_data)
                f.create_dataset("third_masked_data", data=third_masked_data)
                f.create_dataset("first_pol_array", data=pol_array_first)
                f.create_dataset("second_pol_array", data=pol_array_second)
                f.close()
            print("Third approximation", results["xc"], results["yc"])

            ## Display plots

            xr = real_center[0]
            yr = real_center[1]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            pos1 = ax1.imshow(first_masked_data, vmax=7, cmap="jet")
            ax1.scatter(xr, yr, color="lime", label=f"xds:({xr},{yr})")
            ax1.scatter(
                first_center[0],
                first_center[1],
                color="r",
                label=f"calculated center:({first_center[0]},{first_center[1]})",
            )
            ax1.set_title("First approximation: center of mass")
            fig.colorbar(pos1, ax=ax1, shrink=0.6)
            ax1.legend()

            pos2 = ax2.imshow(second_masked_data, vmax=7, cmap="jet")
            ax2.scatter(xr, yr, color="lime", label=f"xds:({xr},{yr})")
            ax2.scatter(
                second_center[0],
                second_center[1],
                color="blueviolet",
                label=f"calculated center:({second_center[0]},{second_center[1]})",
            )
            ax2.set_title("Second approximation: FWHM/R minimization")
            fig.colorbar(pos2, ax=ax2, shrink=0.6)
            ax2.legend()

            pos3 = ax3.imshow(data * mask, vmax=30, cmap="jet")
            ax3.scatter(xr, yr, color="lime", label=f"xds:({xr},{yr})")
            ax3.scatter(
                results["xc"],
                results["yc"],
                color="darkorange",
                label=f'calculated center:({results["xc"]},{results["yc"]})',
            )
            ax3.set_title(
                "Third approximation: Autocorrelation \nof Bragg peaks position"
            )
            fig.colorbar(pos3, ax=ax3, shrink=0.6)
            ax3.legend()

            plt.savefig(f"/home/rodria/Desktop/test/third/lyso_{i}.png")
            plt.close()
            # plt.show()

            ## Display cc matrix plot
            xr = 2 * (real_center[0] - xc)
            yr = 2 * (real_center[1] - yc)
            index = []
            flag = 0
            try:
                index.append(np.where(results["yy"] == yr)[0][0])
                index.append(np.where(results["xx"] == xr)[1][0])
            except IndexError:
                flag = 1
            if (
                results["index"].size != 0
                and results["index"][0] != 0
                and results["index"][1] != 0
                and flag != 1
            ):

                fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
                pos1 = ax1.imshow(results["sub_reduced_cc_matrix"], cmap="jet")
                ax1.scatter(
                    index[0],
                    index[1],
                    color="lime",
                    label=f"xds:({real_center[0]},{real_center[1]})",
                )
                ax1.scatter(
                    results["index"][1],
                    results["index"][0],
                    color="r",
                    label=f'calculated center:({results["xc"]},{results["yc"]})',
                )
                ax1.set_title("Autocorrelation matrix")
                fig.colorbar(pos1, ax=ax1, shrink=0.6)
                ax1.legend()
                plt.savefig(f"/home/rodria/Desktop/test/cc_map/lyso_{i}.png")
                plt.close()


if __name__ == "__main__":
    main()
