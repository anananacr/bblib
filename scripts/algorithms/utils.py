from typing import List, Optional, Callable, Tuple, Any, Dict
from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
import math


def center_of_mass(data: np.ndarray, mask: np.ndarray = None) -> Tuple[int]:
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
    return xc, yc


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
        mask = np.ones((a, b), dtype=bool)
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


def mask_peaks(mask: np.ndarray, indices: tuple, bragg: int) -> np.ndarray:
    """
    Gather coordinates of a box of 3x3 pixels around each point from the indices list. Bragg flag indicates if the mask returned will contain only bragg peaks regions (bragg =1), no bragg peaks regions (bragg=0), or both (bragg =-1).
    Parameters
    ----------
    mask: np.ndarray
        Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.
    indices: tuple
        Bragg peaks coordinates, indices[0] contains x-coordinates of Bragg peaks and indices[1] the corresponding y-coordinates.
    bragg: int
        Bragg flag, choose between return only peaks, only background or both.
    Returns
    ----------
    mask: np.ndarray
        Corresponding mask according to bragg flag choice. It contains zeros for unvalid pixels and one for valid pixels. Mask shape is the same size of data.
    """
    surrounding_positions = []
    count = 0
    for index in zip(indices[0], indices[1]):
        n = 3
        row, col = index
        for i in range(-1 * n, n):
            for k in range(-1 * n, n):
                surrounding_positions.append((row + i, col + k))
        count += 1

    # print(args.bragg)
    if bragg == 1:
        surrounding_mask = np.zeros_like(mask)
        for pos in surrounding_positions:
            row, col = pos
            if 0 <= row < mask.shape[0] and 0 <= col <= mask.shape[1]:
                surrounding_mask[row, col] = 1
    elif bragg == -1:
        surrounding_mask = np.ones_like(mask)
    else:
        surrounding_mask = np.ones_like(mask)
        for pos in surrounding_positions:
            row, col = pos
            if 0 <= row < mask.shape[0] and 0 <= col <= mask.shape[1]:
                surrounding_mask[row, col] = 0

    surrounding_mask[np.where(mask == 0)] = 0
    mask = surrounding_mask
    return mask


def get_format(file_path: str) -> str:
    """
    Return file format with only alphabet letters.
    Parameters
    ----------
    file_path: str

    Returns
    ----------
    extension: str
        File format contanining only alphabetical letters
    """
    ext = (file_path.split("/")[-1]).split(".")[-1]
    filt_ext = ""
    for i in ext:
        if i.isalpha():
            filt_ext += i
    return str(filt_ext)


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


def open_fwhm_map(lines: list, label: str = None):
    """
    Open FWHM/R grid search optmization plot, fit projections in both axis to get the point of maximum sharpness of the radial average.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM/R optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.
    """
    n = int(math.sqrt(len(lines)))

    merged_dict = {}
    for dictionary in lines[:]:

        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z = np.array(merged_dict["fwhm_over_radius"]).reshape((n, n))

    index_y, index_x = np.where(z == np.min(z))
    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 10
    n = z.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax1.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax1.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))

    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("FWHM/R")
    proj_x = np.sum(z, axis=0) / z.shape[0]
    x = np.arange(x[0], x[-1] + 1, 1)

    popt = np.polyfit(x, proj_x, 2)
    residuals = proj_x - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_x - np.mean(proj_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(x[0], x[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax2.plot(
        x_fit,
        y_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Xc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax2.scatter(x, proj_x, color="b")
    ax2.set_ylabel("Average FWHM/R")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("FWHM/R projection in x")
    ax2.legend()
    print(f"xc {round((-1*popt[1])/(2*popt[0]))}")

    proj_y = np.sum(z, axis=1) / z.shape[1]
    x = np.arange(y[0], y[-1] + 1, 1)
    popt = np.polyfit(x, proj_y, 2)
    residuals = proj_y - quadratic(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((proj_y - np.mean(proj_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    x_fit = np.arange(y[0], y[-1] + 0.1, 0.1)
    y_fit = quadratic(x_fit, *popt)
    ax3.plot(
        y_fit,
        x_fit,
        "r",
        label=f"quadratic fit:\nR²: {round(r_squared,5)}, Yc: {round((-1*popt[1])/(2*popt[0]))}",
    )
    ax3.scatter(proj_y, x, color="b")
    ax3.set_xlabel("Average FWHM/R")
    ax3.set_ylabel("yc [px]")
    ax3.set_title("FWHM/R projection in y")
    ax3.legend()
    print(f"yc {round((-1*popt[1])/(2*popt[0]))}")

    fig.colorbar(pos1, ax=ax1, shrink=0.6)

    # Display the figure

    plt.show()
    # plt.savefig(f'/home/rodria/Desktop/fwhm_map/lyso_{label}.png')
    # plt.close()


def fit_fwhm(lines: list) -> Tuple[int]:
    """
    Find minimum of FWHM/R grid search. Fits projections in both axis to get the point of maximum sharpness of the radial average, that will correspond to the center of diffraction.
    Parameters
    ----------
    lines: list
        Output of grid search for FWHM/R optmization, each line should contain a dictionary contaning entries for xc, yc and fwhm_over_radius.

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
    z = np.array(merged_dict["fwhm_over_radius"]).reshape((n, n))

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
