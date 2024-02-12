from typing import List, Optional, Callable, Tuple, Any, Dict
from numpy import exp
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.colors as colors
import math
import json
import pandas as pd
import sys
from scipy import ndimage


def center_of_mass(data: np.ndarray, mask: np.ndarray = None) -> List[int]:
    """
    Adapted from Robert Bücker work on diffractem (https://github.com/robertbuecker/diffractem/tree/master)
    Bücker, R., Hogan-Lamarre, P., Mehrabi, P. et al. Serial protein crystallography in an electron microscope. Nat Commun 11, 996 (2020). https://doi.org/10.1038/s41467-020-14793-0

    Parameters
    ----------
    data: np.ndarray
        Input data in which center of mass will be calculated. Values equal or less than zero will not be considered.
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.

    Returns
    ----------
    xc, yc: int
         coordinates of the diffraction center in x and y, such as the image center corresponds to data[yc, xc].
    """

    if mask is None:
        mask = np.ones_like(data)
    data = data * mask
    indices = np.where(data > 0)
    xc = np.sum(data[indices] * indices[1]) / np.sum(data[indices])
    yc = np.sum(data[indices] * indices[0]) / np.sum(data[indices])

    if np.isnan(xc) or np.isnan(yc):
        converged = 0
    else:
        converged = 1

    return converged, [xc, yc]


def azimuthal_average(
    data: np.ndarray, center: tuple = None, mask: np.ndarray = None
) -> np.ndarray:
    """
    Calculate azimuthal integration of data in relation to the center of the image
    Adapted from L. P. René de Cotret work on scikit-ued (https://github.com/LaurentRDC/scikit-ued/tree/master)
    L. P. René de Cotret, M. R. Otto, M. J. Stern. and B. J. Siwick, An open-source software ecosystem for the interactive exploration of ultrafast electron scattering data, Advanced Structural and Chemical Imaging 4:11 (2018) DOI: 10.1186/s40679-018-0060-y.

    Parameters
    ----------
    data: np.ndarray
        Input data in which center of mass will be calculated. Values equal or less than zero will not be considered.
    center: tuple
        Center coordinates of the radial average (xc, yc)->(col, row).
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.
    Returns
    ----------
    radius: np.ndarray
        radial axis radius in pixels

    intensity: np.ndarray
        Integrated intensity normalized by the number of valid pixels
    """
    a = data.shape[0]
    b = data.shape[1]
    if mask is None:
        mask = np.zeros((a, b), dtype=bool)
    else:
        mask.astype(bool)

    if center is None:
        center = [b / 2, a / 2]
    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    Rint = np.rint(R).astype(int)

    valid = mask.flatten()
    data = data.flatten()
    Rint = Rint.flatten()

    px_bin = np.bincount(Rint, weights=valid * data)
    r_bin = np.bincount(Rint, weights=valid)
    radius = np.arange(0, r_bin.size)
    # Replace by one if r_bin is zero for division
    np.maximum(r_bin, 1, out=r_bin)

    return radius, px_bin / r_bin


def correct_polarization_python(
    x: np.ndarray,
    y: np.ndarray,
    dist: float,
    data: np.ndarray,
    mask: np.ndarray,
    polarization_axis: str = "x",
    p: float = 0.99,
) -> np.ndarray:
    """
    Correct data for polarisation effect, version in Python. It is based on pMakePolarisationArray from https://github.com/galchenm/vdsCsPadMaskMaker/blob/main/new-versions/maskMakerGUI-v2.py#L234
    Acknowledgements: Oleksandr Yefanov, Marina Galchenkova

    Parameters
    ----------
    x: np.ndarray
        Array containg pixels coordinates in x (pixels) distance from the direct beam. It has same shape of data.
    y: np.ndarray
        Array containg pixels coordinates in y (pixels) distance from the direct beam. It has same shape of data.
    dist: float
        z distance coordinates of the detector position in PIXELS.
    data: np.ndarray
        Raw data frame in which polarization correction will be applied.
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.

    Returns
    ----------
    corrected_data: np.ndarray
        Corrected data frame for polarization effect.
    pol: np.ndarray
        Polarization array for polarization correction.
    """

    mask = mask.astype(bool)
    mask = mask.flatten()
    Int = np.reshape(data.copy(), len(mask))
    pol = mask.copy().astype(np.float32)
    pol = make_polarization_array(pol, x.flatten(), y.flatten(), dist, p)
    Int = Int / pol
    return Int.reshape(data.shape), pol.reshape(data.shape)


def make_polarization_array(
    pol: np.ndarray, cox: np.ndarray, coy: np.ndarray, detdist: float, poldegree: float
) -> np.ndarray:
    """
    Create the polarization array for horizontal polarization correction, version in Python. It is based on pMakePolarisationArray from https://github.com/galchenm/vdsCsPadMaskMaker/blob/main/new-versions/maskMakerGUI-v2.py#L234
    Acknowledgements: Oleksandr Yefanov, Marina Galchenkova

    Parameters
    ----------
    pol: np.ndarray
        An array where polarization arra will be built based on its shape. Mask shape is the same size of data. Unvalid pixels (values containing 0) will be skipped from calculation and put 1.
    cox: np.ndarray
        Array containg pixels coordinates in x (pixels) distance from the direct beam. It has same shape of data.
    coy: np.ndarray
        Array containg pixels coordinates in y (pixels) distance from the direct beam. It has same shape of data.
    detdist: float
        Detector distance from the sample in meters . The detctor distance will be transformed in pixel units based on Res defined as global parameter.
    poldegree: float
        Polarization degree, horizontal polarization at DESY p=0.99.
    Returns
    ----------
    pol: np.ndarray
        Polarization array for polarization correction.
    """

    z = detdist * np.ones(cox.shape[0])
    valid = np.where(pol == 1)

    pol[valid] = 1 - (
        (poldegree * (cox[valid] ** 2) + (1 - poldegree) * (coy[valid] ** 2))
        / (cox[valid] ** 2 + coy[valid] ** 2 + z[valid] ** 2)
    )
    pol[np.where(pol == 0)] = 1.0

    return pol


def update_corner_in_geom(geom: str, new_xc: float, new_yc: float):
    """
    Write new direct beam position in detector coordinates in the geometry file.

    Parameters
    ----------
    geom: str
        CrystFEL eometry file name to be updated .geom format.
    new_xc: float
        Direct beam position in detector coordinates in the x axis.
    new_yc: float
        Direct beam position in detector coordinates in the y axis.

    Returns
    ----------
    corrected_data: np.ndarray
        Corrected data frame for polarization effect.
    pol: np.ndarray
        Polarization array for polarization correction.
    """
    # convert y x values to i j values
    y = -1 * new_yc
    x = -1 * new_xc

    f = open(geom, "r")
    lines = f.readlines()
    f.close()

    new_lines = []

    for i in lines:
        key_args = i.split(" = ")[0]

        if key_args[-8:] == "corner_x":
            new_lines.append(f"{key_args} = {x}\n")
        elif key_args[-8:] == "corner_y":
            new_lines.append(f"{key_args} = {y}\n")
        else:
            new_lines.append(i)

    f = open(geom, "w")
    for i in new_lines:
        f.write(i)
    f.close()


def mask_peaks(mask: np.ndarray, indices: tuple, bragg: int, n: int) -> np.ndarray:
    """
    Gather coordinates of a box of 1x1 pixels around each point from the indices list. Bragg flag indicates if the mask returned will contain only bragg peaks regions (bragg =1), no bragg peaks regions (bragg=0), or both (bragg =-1).
    Parameters
    ----------
    mask: np.ndarray
        An array where mask will be built based on its shape. Mask shape is the same size of data.
    indices: tuple
        Bragg peaks coordinates, indices[0] contains x-coordinates of Bragg peaks and indices[1] the corresponding y-coordinates.
    bragg: int
        Bragg flag, choose between return only peaks, only background or both (bypass masking of peaks).
    n: int
        n pixels to build a 2n box around the peak.
    Returns
    ----------
    surrounding_mask: np.ndarray
        Corresponding mask according to bragg flag choice. It contains zeros for unvalid pixels and one for valid pixels. Mask shape is the same size of data.
    """
    surrounding_positions = []
    count = 0
    for index in zip(indices[0], indices[1]):
        row, col = index
        for i in range(-n, n + 1):
            for k in range(-n, n + 1):
                surrounding_positions.append((row + i, col + k))
        count += 1

    # print(args.bragg)
    if bragg == 1:
        surrounding_mask = np.zeros_like(mask)
        for pos in surrounding_positions:
            row, col = pos
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
                surrounding_mask[row, col] = 1
    elif bragg == -1:
        surrounding_mask = np.ones_like(mask)
    else:
        surrounding_mask = np.ones_like(mask)
        for pos in surrounding_positions:
            row, col = pos
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
                surrounding_mask[row, col] = 0

    return surrounding_mask


def gaussian(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """
    Gaussian function.

    Parameters
    ----------
    x: np.ndarray
        x array of the spectrum.
    a, x0, sigma: float
        gaussian parameters

    Returns
    ----------
    y: np.ndarray
        value of the function evaluated
    """
    return a * exp(-((x - x0) ** 2) / (2 * sigma**2))


def gaussian_lin(
    x: np.ndarray, a: float, x0: float, sigma: float, m: float, n: float
) -> np.ndarray:
    """
    Gaussian function.

    Parameters
    ----------
    x: np.ndarray
        x array of the spectrum.
    a, x0, sigma: float
        gaussian parameters

    Returns
    ----------
    y: np.ndarray
        value of the function evaluated
    """
    return m * x + n + a * exp(-((x - x0) ** 2) / (2 * sigma**2))


def quadratic(x, a, b, c):
    """
    Quadratic function.

    Parameters
    ----------
    x: np.ndarray
        x array of the spectrum.
    a, b, c: float
        quadratic parameters

    Returns
    ----------
    y: np.ndarray
        value of the function evaluated
    """
    return a * x**2 + b * x + c


def open_fwhm_map(lines: list, output_folder: str, label: str, pixel_step: int):
    """
    Open FWHM grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm"]).reshape((n, n))
    r = np.array(merged_dict["r_squared"]).reshape((n, n))

    index_y, index_x = np.where(z == np.min(z))
    pos1 = ax1.imshow(z, cmap="YlGnBu")
    step = 10
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax1.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax1.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("FWHM")

    pos2 = ax2.imshow(r, cmap="YlGnBu")
    step = 10
    n = z.shape[0]
    ax2.set_xticks(np.arange(0, n, step, dtype=int))
    ax2.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax2.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax2.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("R²")

    proj_x = np.sum(z, axis=0)
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)

    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(x[0], x[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax3.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Xc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax3.scatter(x, proj_x, color="b")
    ax3.set_ylabel("Average FWHM")
    ax3.set_xlabel("xc [px]")
    ax3.set_title("FWHM projection in x")
    ax3.legend()
    xc = round((-1 * popt[1]) / (2 * popt[0]))
    # print(f"xc {round((-1*popt[1])/(2*popt[0]))}")

    proj_y = np.sum(z, axis=1)
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(y[0], y[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax4.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Yc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax4.scatter(x, proj_y, color="b")
    ax4.set_ylabel("Average FWHM")
    ax4.set_xlabel("yc [px]")
    ax4.set_title("FWHM projection in y")
    ax4.legend()
    yc = round((-1 * popt[1]) / (2 * popt[0]))
    # print(f"yc {round((-1*popt[1])/(2*popt[0]))}")

    fig.colorbar(pos1, ax=ax1, shrink=0.6)
    fig.colorbar(pos2, ax=ax2, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f"{output_folder}/plots/fwhm_map/{label}.png")
    plt.close()
    if r_squared > 0.7:
        return xc, yc
    else:
        return False


def open_fwhm_map_global_min(
    lines: list, output_file: str, pixel_step: int, PlotsFlag: bool
):
    """
    Open FWHM grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm"], dtype=np.float64).reshape((n, n))
    r = np.array(merged_dict["r_squared"]).reshape((n, n))
    z = np.nan_to_num(z)
    r = np.nan_to_num(r)
    pos1 = ax1.imshow(z, cmap="YlGnBu", vmax=50)
    step = 10
    n = z.shape[0]

    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))

    ticks_len = (np.arange(0, n, step)).shape[0]
    step = round(step * (abs(x[0] - x[1])), 1)
    ax1.set_xticklabels(
        np.linspace(round(x[0], 1), round(x[-1] + step, 1), ticks_len, dtype=int),
        rotation=45,
    )
    ax1.set_yticklabels(
        np.linspace(round(y[0], 1), round(y[-1] + step, 1), ticks_len, dtype=int)
    )

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("FWHM")

    pos2 = ax2.imshow(r, cmap="YlGnBu")
    step = 10
    n = z.shape[0]

    ax2.set_xticks(np.arange(0, n, step, dtype=int))
    ax2.set_yticks(np.arange(0, n, step, dtype=int))

    ticks_len = (np.arange(0, n, step)).shape[0]
    step = round(step * (abs(x[0] - x[1])), 1)
    ax2.set_xticklabels(
        np.linspace(round(x[0], 1), round(x[-1] + step, 1), ticks_len, dtype=int),
        rotation=45,
    )
    ax2.set_yticklabels(
        np.linspace(round(y[0], 1), round(y[-1] + step, 1), ticks_len, dtype=int)
    )

    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("R²")

    proj_x = np.sum(z, axis=0) // n
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)
    index_x = np.unravel_index(np.argmin(proj_x, axis=None), proj_x.shape)
    # print(index_x)
    xc = x[index_x]
    ax3.scatter(x, proj_x, color="b")
    ax3.scatter(xc, proj_x[index_x], color="r", label=f"xc: {xc}")
    ax3.set_ylabel("Average FWHM")
    ax3.set_xlabel("xc [px]")
    ax3.set_title("FWHM projection in x")
    ax3.legend()

    proj_y = np.sum(z, axis=1) // n
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    index_y = np.unravel_index(np.argmin(proj_y, axis=None), proj_y.shape)
    yc = x[index_y]
    ax4.scatter(x, proj_y, color="b")
    ax4.scatter(yc, proj_y[index_y], color="r", label=f"yc: {yc}")
    ax4.set_ylabel("Average FWHM")
    ax4.set_xlabel("yc [px]")
    ax4.set_title("FWHM projection in y")
    ax4.legend()

    fig.colorbar(pos1, ax=ax1, shrink=0.6)
    fig.colorbar(pos2, ax=ax2, shrink=0.6)

    # Display the figure

    # plt.show()
    if PlotsFlag:
        plt.savefig(f"{output_file}.png")
    plt.close()
    return xc, yc


def open_r_sqrd_map_global_max(
    lines: list, output_file: str, pixel_step: int, PlotsFlag: bool
):
    """
    Open FWHM grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm"], dtype=np.float32).reshape((n, n))
    r = np.array(merged_dict["r_squared"], dtype=np.float32).reshape((n, n))
    # r = ndimage.median_filter(r, 10)

    z = np.nan_to_num(z)
    r = np.nan_to_num(r)

    pos1 = ax1.imshow(z, cmap="YlGnBu")
    step = 10
    n = z.shape[0]

    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))

    ticks_len = (np.arange(0, n, step)).shape[0]
    step = round(step * (abs(x[0] - x[1])), 1)
    ax1.set_xticklabels(
        np.linspace(round(x[0], 1), round(x[-1] + step, 1), ticks_len, dtype=int),
        rotation=45,
    )
    ax1.set_yticklabels(
        np.linspace(round(y[0], 1), round(y[-1] + step, 1), ticks_len, dtype=int)
    )

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("FWHM")

    pos2 = ax2.imshow(r, cmap="YlGnBu")
    step = 10
    n = z.shape[0]

    ax2.set_xticks(np.arange(0, n, step, dtype=int))
    ax2.set_yticks(np.arange(0, n, step, dtype=int))

    ticks_len = (np.arange(0, n, step)).shape[0]
    step = round(step * (abs(x[0] - x[1])), 1)
    ax2.set_xticklabels(
        np.linspace(round(x[0], 1), round(x[-1] + step, 1), ticks_len, dtype=int),
        rotation=45,
    )
    ax2.set_yticklabels(
        np.linspace(round(y[0], 1), round(y[-1] + step, 1), ticks_len, dtype=int)
    )

    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("R²")

    proj_x = np.sum(r, axis=0)
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)
    index_x = np.unravel_index(np.argmax(proj_x, axis=None), proj_x.shape)
    # print(index_x)
    xc = x[index_x]
    ax3.scatter(x, proj_x, color="b")
    ax3.scatter(xc, proj_x[index_x], color="r", label=f"xc: {xc}")
    ax3.set_ylabel("Average FWHM")
    ax3.set_xlabel("xc [px]")
    ax3.set_title("FWHM projection in x")
    ax3.legend()

    proj_y = np.sum(r, axis=1)
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    index_y = np.unravel_index(np.argmax(proj_y, axis=None), proj_y.shape)
    yc = x[index_y]
    ax4.scatter(x, proj_y, color="b")
    ax4.scatter(yc, proj_y[index_y], color="r", label=f"yc: {yc}")
    ax4.set_ylabel("Average FWHM")
    ax4.set_xlabel("yc [px]")
    ax4.set_title("FWHM projection in y")
    ax4.legend()

    fig.colorbar(pos1, ax=ax1, shrink=0.6)
    fig.colorbar(pos2, ax=ax2, shrink=0.6)

    # Display the figure

    # plt.show()
    if PlotsFlag:
        plt.savefig(f"{output_file}.png")
    plt.close()
    return xc, yc


def open_distance_map_global_min(
    lines: list, output_folder: str, label: str, pixel_step: int, plots_flag: bool
) -> tuple:
    """
    Open distance minimization plot, fit projections in both axis to get the point of minimum distance.

    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
    """

    n = int(math.sqrt(len(lines)))
    pixel_step /= 2
    merged_dict = {}
    for dictionary in lines[:]:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["d"], dtype=np.float64).reshape((n, n))
    z = np.nan_to_num(z)

    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 20
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=float))
    ax1.set_yticks(np.arange(0, n, step, dtype=float))
    step = round(step * (abs(x[0] - x[1])), 1)
    ax1.set_xticklabels(
        np.arange(round(x[0], 1), round(x[-1] + step, 1), step, dtype=int), rotation=45
    )
    ax1.set_yticklabels(
        np.arange(round(y[0], 1), round(y[-1] + step, 1), step, dtype=int)
    )

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("Distance [px]")

    proj_x = np.sum(z, axis=0)
    # print('proj',len(proj_x))
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)
    # print('x',len(x))
    index_x = np.unravel_index(np.argmin(proj_x, axis=None), proj_x.shape)
    # print(index_x)
    xc = x[index_x]
    ax2.scatter(x, proj_x + pixel_step, color="b")
    ax2.scatter(xc, proj_x[index_x], color="r", label=f"xc: {np.round(xc,1)}")
    ax2.set_ylabel("Average distance [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("Distance projection in x")
    ax2.legend()

    proj_y = np.sum(z, axis=1)
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    index_y = np.unravel_index(np.argmin(proj_y, axis=None), proj_y.shape)
    yc = x[index_y]
    ax3.scatter(x, proj_y, color="b")
    ax3.scatter(yc, proj_y[index_y], color="r", label=f"yc: {np.round(yc,1)}")
    ax3.set_ylabel("Average Distance [px]")
    ax3.set_xlabel("yc [px]")
    ax3.set_title("Distance projection in y")
    ax3.legend()

    fig.colorbar(pos1, ax=ax1, shrink=0.6)

    # Display the figure

    # plt.show()

    if int(np.sum(proj_y)) == 0 or int(np.sum(proj_x)) == 0:
        converged = 0
    else:
        converged = 1
        if plots_flag:
            plt.savefig(f"{output_folder}/distance_map/{label}.png")

    plt.close()
    return xc, yc, converged


def open_distance_map_fit_min(
    lines: list, output_folder: str, label: str, pixel_step: int
) -> tuple:
    """
    Open distance minimization plot, fit projections in both axis to get the point of minimum distance.

    Parameters
    ----------
    lines: list
        Output of grid search for distance optmization, each line should contain a dictionary contaning entries for xc, yc and distance.
    """

    n = int(math.sqrt(len(lines)))
    pixel_step /= 2
    merged_dict = {}
    for dictionary in lines[:]:

        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["d"], dtype=np.float64).reshape((n, n))

    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 20
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=float))
    ax1.set_yticks(np.arange(0, n, step, dtype=float))

    ticks_len = (np.arange(0, n, step)).shape[0]
    step = round(step * (abs(x[0] - x[1])), 1)

    ax1.set_xticklabels(
        np.linspace(round(x[0], 1), round(x[-1] + step, 1), ticks_len, dtype=int),
        rotation=45,
    )
    ax1.set_yticklabels(
        np.linspace(round(y[0], 1), round(y[-1] + step, 1), ticks_len, dtype=int)
    )

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("Distance [px]")

    proj_x = np.sum(z, axis=0) / z.shape[0]
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)

    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(x[0], x[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)

    a, b, c = popt
    xc = round((-1 * b) / (2 * a), 1)
    y_min = round((-1 * (b**2) + 4 * a * c) / (4 * a), 1)
    ax2.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, xc: {xc}",
    )

    ax2.scatter(x, proj_x + pixel_step, color="b")
    ax2.scatter(xc, y_min, color="r", label=f"xc: {xc}")
    ax2.set_ylabel("Average distance (px)")
    ax2.set_xlabel("Detector center in x (px)")
    ax2.set_title("Distance projection in x")
    ax2.legend()

    proj_y = np.sum(z, axis=1) / z.shape[1]
    x = np.arange(y[0], y[-1] + pixel_step, pixel_step)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    a, b, c = popt
    yc = round((-1 * b) / (2 * a), 1)
    y_min = round((-1 * (b**2) + 4 * a * c) / (4 * a), 1)

    x_fit = np.arange(y[0], y[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax3.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, yc: {yc}",
    )

    ax3.scatter(x, proj_y, color="b")
    ax3.scatter(yc, y_min, color="r", label=f"yc: {yc}")
    ax3.set_ylabel("Average distance (px)")
    ax3.set_xlabel("Detector center in y (px)")
    ax3.set_title("Distance projection in y")
    ax3.legend()

    fig.colorbar(pos1, ax=ax1, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f"{output_folder}/distance_map_fit/{label}.png")
    plt.close()
    if int(np.sum(proj_y)) == 0 or int(np.sum(proj_x)) == 0:
        converged = 0
    else:
        converged = 1
    return xc, yc, converged


def fit_fwhm(lines: list, pixel_step: int) -> Tuple[int]:
    """
    Find minimum of FWHM grid search. Fits projections in both axis to get the point of maximum sharpness of the radial average, that will correspond to the center of diffraction.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.

    Returns
    ----------
    xc, yc: int
        coordinates of the diffraction center in x and y, such as the image center corresponds to data[yc, xc].
    """
    n = int(math.sqrt(len(lines)))
    merged_dict = {}
    for dictionary in lines[:]:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm"]).reshape((n, n))

    proj_x = np.sum(z, axis=0) / z.shape[0]
    x = np.arange(x[0], x[-1] + 1, 1)
    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared_x = 1 - (ss_res / ss_tot)
    xc = round((-1 * popt[1]) / (2 * popt[0]))

    proj_y = np.sum(z, axis=1) / z.shape[1]
    x = np.arange(y[0], y[-1] + 1, 1)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared_y = 1 - (ss_res / ss_tot)
    yc = round((-1 * popt[1]) / (2 * popt[0]))

    return xc, yc


def shift_image_by_n_pixels(data: np.ndarray, n: int, axis: int) -> np.ndarray:
    """
    Linear translation of image by n pixels in given axis. Empty values in the shifted image is filled with zero.

    Parameters
    ----------
    data: np.ndarray
        Input image to be shifted
    n: int
        Number of pixels to be shifted.
    axis: int
        Axis in which the image will be shifted. Axis 0 corresponds to a shift in the rows (y-axis), axis 1 shifts in the columns (x-axis).
    Returns
    ----------
    shifted_data: np.ndarray
        Data shifted by n pixels in axis.
    """
    max_row, max_col = data.shape
    # print(max_row,max_col)
    if axis == 1 and n >= 0:
        shifted_data = np.pad(data, pad_width=[(0, 0), (abs(n), 0)], mode="constant")
        image_cut = shifted_data[:max_row, :max_col]
    elif axis == 1 and n < 0:
        shifted_data = np.pad(data, pad_width=[(0, 0), (0, abs(n))], mode="constant")
        image_cut = shifted_data[:max_row, abs(n) :]
    elif axis == 0 and n >= 0:
        shifted_data = np.pad(data, pad_width=[(abs(n), 0), (0, 0)], mode="constant")
        image_cut = shifted_data[:max_row, :max_col]
    elif axis == 0 and n < 0:
        shifted_data = np.pad(data, pad_width=[(0, abs(n)), (0, 0)], mode="constant")
        image_cut = shifted_data[abs(n) :, :max_col]
    # print("Image cut shape", image_cut.shape)
    return image_cut


def table_of_center(
    crystal: int, rot: int, center_file: str = None, loaded_table_center: Dict = None
) -> List[int]:
    """
    Return theoretical center positions for the data given its ID (crystal and rotation number) in a .txt file.

    Parameters
    ----------
    crystal: int
        Crystal number identification.
    rot: int
        Rotation number identification.
    center_file:
        Path to the theoretical center positions .txt file.
        Example: center.txt
        {'crystal': 1, 'rot': 1, 'center_x': 831, 'center_y': 993}
        {'crystal': 1, 'rot': 2, 'center_x': 834, 'center_y': 982}
    loaded_table_center: Dict
        Bypass loading of the table if the function had already been called.

    Returns
    ----------
    center_theory: Tuple[int]
        Theoretical center positions for the data with given crystal and rotation ID.
    """

    if loaded_table_center is None:
        if center_file is None:
            data = {
                "crystal": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
                "rot": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                "center_x": [
                    831,
                    834,
                    825,
                    830,
                    831,
                    832,
                    832,
                    833,
                    831,
                    831,
                    831,
                    834,
                    831,
                    833,
                    831,
                    829,
                    826,
                    825,
                    823,
                    831,
                ],
                "center_y": [
                    993,
                    982,
                    979,
                    973,
                    962,
                    928,
                    927,
                    925,
                    894,
                    885,
                    877,
                    851,
                    833,
                    824,
                    810,
                    795,
                    785,
                    774,
                    766,
                    761,
                ],
            }
        else:
            # print(center_file)
            data = get_table_center(center_file)

            # print(data)
        loaded_table_center = data.copy()

    data = loaded_table_center
    df = pd.DataFrame.from_dict(data)
    # print(df)
    match = df.loc[(df["crystal"] == crystal) & (df["rot"] == rot)].reset_index()

    return [match["center_x"][0], match["center_y"][0]], loaded_table_center


def get_table_center(center_file: str) -> Dict:
    """
    Load theoretical center positions for the data given its ID (crystal and rotation number) from a .txt file.

    Parameters
    ----------
    center_file:
        Path to the theoretical center positions .txt file.
        Example: center.txt
        {'crystal': 1, 'rot': 1, 'center_x': 831, 'center_y': 993}
        {'crystal': 1, 'rot': 2, 'center_x': 834, 'center_y': 982}

    Returns
    ----------
    loaded_table_center: Dict
        Theoretical center positions table.
    """
    data = open(center_file, "r").read().splitlines()
    data = [x.replace("'", '"') for x in data]
    data = [json.loads(d) for d in data]
    # print(data)
    return transpose_dict(data)


def transpose_dict(data: list) -> dict:
    """
    Transposes a list of dictionaries into a dictionary of lists.

    Parameters:
        data (list): A list of dictionaries to be transposed.

    Returns:
        dict: A dictionary with keys from the original dictionaries and values as lists
              containing the corresponding values from each dictionary.

    Example:
        >>> data = [{'key1': 1, 'key2': 2}, {'key1': 3, 'key2': 4}]
        >>> transpose_dict(data)
        {'key1': [1, 3], 'key2': [2, 4]}
    """
    result = {}
    for d in data:
        for k, v in d.items():
            if k not in result:
                result[k] = []
            result[k].append(v)

    return result


def get_center_theory(
    files_path: np.ndarray, center_file: str = None, loaded_table_center: str = None
) -> List[int]:
    """
    Extract crystal and rotation number ID from the file name and get theoretical center positions from a .txt file.

    Parameters
    ----------
    files_path: np.ndarray
        Array of input images path.
    center_file:
        Path to the theoretical center positions .txt file.
        Example: center.txt
        {'crystal': 1, 'rot': 1, 'center_x': 831, 'center_y': 993}
        {'crystal': 1, 'rot': 2, 'center_x': 834, 'center_y': 982}
    loaded_table_center: Dict
        Theoretical center positions table.
    Returns
    ----------
    center_theory: List[int]
        Theoretical center positions table for input images.
    loaded_table_center: Dict
        Theoretical center positions table from .txt file to avoid opening it many times.
    """
    center_theory = []
    for i in files_path:
        label = str(i).split("/")[-1]
        crystal = int(label.split("_")[-3][-2:])
        rot = int(label.split("_")[-2][:])
        center, loaded_table_center = table_of_center(
            crystal, rot, center_file, loaded_table_center
        )
        center_theory.append(center)
    center_theory = np.array(center_theory)
    return center_theory, loaded_table_center


def fill_gaps(data: np.ndarray, center: tuple, mask: np.ndarray) -> np.ndarray:
    ave_r, ave_y = azimuthal_average(data, center, mask)
    filled_data = (data * mask).copy().astype(np.float32)
    y, x = np.where(data * mask <= 0)

    for i in zip(y, x):
        radius = math.sqrt((i[0] - center[1]) ** 2 + (i[1] - center[0]) ** 2)

        index = np.where(ave_r == round(radius))[0]
        if not index:
            index = ave_r[0:10]

        filled_data[i] = np.mean(ave_y[index])
    return filled_data


def circle_mask(data: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    """
    Make a  ring mask for the data

    Parameters
    ----------
    data: np.ndarray
        Image in which mask will be shaped
    radius: int
        Outer radius of the mask

    Returns
    ----------
    mask: np.ndarray
    """

    bin_size = bin
    a = data.shape[0]
    b = data.shape[1]

    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    return (np.greater(R, radius)).astype(np.int32)


def ring_mask(data, center, inner_radius, outer_radius):
    """
    Make a  ring mask for the data

    Parameters
    ----------
    data: np.ndarray
        Image in which mask will be shaped
    center: tuple (xc,yc)

    inner_radius: int

    outer_radius: int
    Returns
    ----------
    mask: np.ndarray
    """

    bin_size = bin
    a = data.shape[0]
    b = data.shape[1]
    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    bin_size = outer_radius - inner_radius
    return np.greater(R, outer_radius - bin_size) & np.less(R, outer_radius + bin_size)
