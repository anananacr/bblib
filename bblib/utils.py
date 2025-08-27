"""
This module defines auxiliary funtions to process the data.
"""

from numpy import exp
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pathlib
import math

plt.switch_backend("agg")


def center_of_mass(data: np.ndarray, mask: np.ndarray = None) -> list[int]:
    """
    Adapted from Robert Bücker work on diffractem (https://github.com/robertbuecker/diffractem/tree/master)
    Bücker, R., Hogan-Lamarre, P., Mehrabi, P. et al. Serial protein crystallography in an electron microscope. Nat Commun 11, 996 (2020). https://doi.org/10.1038/s41467-020-14793-0

    Args:
        data (np.ndarray): Input data in which center of mass will be calculated. Values equal or less than zero will not be considered.
        mask (np.ndarray): Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.

    Returns:
        xc (int): Coordinate of the diffraction center in x, such that the image center corresponds to data [yc, xc].
        yc (int): Coordinate of the diffraction center in y, such that the image center corresponds to data [yc, xc].
    """

    if mask is None:
        mask = np.ones_like(data)
    data = data * mask
    indexes = np.where(data > 0)
    if np.sum(data[indexes]) > 1e-7:
        xc = np.sum(data[indexes] * indexes[1]) / np.sum(data[indexes])
        yc = np.sum(data[indexes] * indexes[0]) / np.sum(data[indexes])
    else:
        xc = -1
        yc = -1

    if np.isnan(xc) or np.isnan(yc):
        xc = -1
        yc = -1

    return [np.round(xc, 1), np.round(yc, 1)]


@njit
def _radial_reduce(vals, rr, maxr):
    sums = np.zeros(maxr + 1, dtype=np.float64)
    counts = np.zeros(maxr + 1, dtype=np.int64)
    for i in range(rr.size):
        r = rr[i]
        sums[r] += vals[i]
        counts[r] += 1
    return sums, counts


def _precompute_rbins(shape, center):
    a, b = shape
    yy, xx = np.ogrid[:a, :b]
    dy = yy - center[1]
    dx = xx - center[0]
    R = np.hypot(dx, dy)
    Rint = (R + 0.5).astype(np.int32)
    maxr = int(Rint.max())
    return Rint, maxr


def azimuthal_average_fast(
    data: np.ndarray, center: tuple = None, mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns intensity over radius, where intensity is the mean per integer radius.
    Improve performance.

    Args:
        data (np.ndarray): Input data in which center of mass will be calculated. Values equal or less than zero will not be considered.
        center (tuple): Center coordinates of the radial average (xc, yc)->(col, row).
        mask (np.ndarray): Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.

    Returns:
        radius (np.ndarray): Radial axis in pixels.
        intensity (np.ndarray): Integrated intensity normalized by the number of valid pixels.
    """
    a, b = data.shape
    if center is None:
        center = (b / 2, a / 2)

    if mask is None:
        mask = np.ones((a, b), dtype=bool)
    else:
        mask = mask.astype(bool, copy=False)

    Rint, maxr = _precompute_rbins(data.shape, (float(center[0]), float(center[1])))

    m = mask.ravel()
    rr = Rint.ravel()[m]
    vals = data.ravel()[m]

    sums, counts = _radial_reduce(vals, rr, maxr)

    with np.errstate(invalid="ignore", divide="ignore"):
        prof = sums / np.maximum(counts, 1)

    radius = np.arange(prof.size, dtype=np.int32)
    return radius, prof


def correct_polarization(
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

    Args:
        x (np.ndarray): Array containg pixels coordinates in x (pixels) distance from the direct beam. It has same shape of data.
        y (np.ndarray): Array containg pixels coordinates in y (pixels) distance from the direct beam. It has same shape of data.
        dist (float): z distance coordinates of the detector position in pixels.
        data (np.ndarray): Raw data frame in which polarization correction will be applied.
        mask (np.ndarray): Corresponding mask of data, containing zeros for unvalid pixels and one for valid pixels. Mask shape should be same size of data.
        p (float): Polarization degree.
    Returns:
        corrected_data (np.ndarray): Corrected data frame for polarization effect.
        pol (np.ndarray): Polarization array for polarization correction.
    """

    mask = mask.astype(bool)
    mask = mask.flatten()
    intensity = np.reshape(data.copy(), len(mask))
    pol = mask.copy().astype(np.float32)
    if polarization_axis == "x":
        pol = make_polarization_array(pol, x.flatten(), y.flatten(), dist, p)
    elif polarization_axis == "y":
        pol = make_polarization_array(pol, x.flatten(), y.flatten(), dist, 1 - p)
    else:
        raise ValueError("Unreconized polarization axis. Options available are x or y.")

    intensity = intensity / pol
    return intensity.reshape(data.shape), pol.reshape(data.shape)


def make_polarization_array(
    pol: np.ndarray, cox: np.ndarray, coy: np.ndarray, detdist: float, poldegree: float
) -> np.ndarray:
    """
    Create the polarization array for horizontal polarization correction, version in Python. It is based on pMakePolarisationArray from https://github.com/galchenm/vdsCsPadMaskMaker/blob/main/new-versions/maskMakerGUI-v2.py#L234
    Acknowledgements: Oleksandr Yefanov, Marina Galchenkova

    Args:
        pol (np.ndarray): An array where polarization array will be built based on its shape. Mask shape is the same size of data. Unvalid pixels (values containing 0) will be skipped from calculation and put 1.
        cox (np.ndarray): Array containg pixels coordinates in x (pixels) distance from the direct beam. It has same shape of data.
        coy (np.ndarray): Array containg pixels coordinates in y (pixels) distance from the direct beam. It has same shape of data.
        detdist (float): Detector distance from the sample in meters . The detctor distance will be transformed in pixel units based on Res defined as global parameter.
        poldegree (float): Polarization degree from [0,1]. If the polarization is completely horizontal (along the x-axis), then poldegree equals 1.
    Returns:
        pol (np.ndarray): Polarization array for polarization correction.
    """

    z = detdist * np.ones(cox.shape[0])
    valid = np.where(pol == 1)

    pol[valid] = 1 - (
        (poldegree * (cox[valid] ** 2) + (1 - poldegree) * (coy[valid] ** 2))
        / (cox[valid] ** 2 + coy[valid] ** 2 + z[valid] ** 2)
    )
    pol[np.where(pol == 0)] = 1.0

    return pol


def mask_peaks(mask: np.ndarray, indexes: tuple, bragg: int, n: int) -> np.ndarray:
    """
    Gather coordinates of a box of 1x1 pixels around each point from the indexes list. Bragg flag indicates if the mask returned will contain only bragg peaks regions (bragg =1), no bragg peaks regions (bragg=0), or both (bragg =-1).

    Args:
        mask (np.ndarray): An array where mask will be built based on its shape. Mask shape is the same size of data.
        indexes (tuple): Bragg peaks coordinates, indexes[0] contains x-coordinates of Bragg peaks and indexes[1] the corresponding y-coordinates.
        bragg (int): Bragg flag, choose between return only peaks, only background or both (bypass masking of peaks).
        n (int): Number of pixels to build a 2*n box around the Bragg peaks.

    Returns:
        surrounding_mask (np.ndarray): Corresponding mask according to bragg flag choice. It contains zeros for unvalid pixels and one for valid pixels. Mask shape is the same size of data.
    """
    surrounding_positions = []
    count = 0
    for index in zip(indexes[0], indexes[1]):
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


def gaussian_lin(
    x: np.ndarray, a: float, x0: float, sigma: float, m: float, n: float
) -> np.ndarray:
    """
    Gaussian function summed to a linear function.

    Args:
        x (np.ndarray): x-axis.
        a (float): Amplitude of the Gaussian.
        x0 (float): Average of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
        m: Angular coefficient.
        n: Linear coefficient.

    Returns:
        y (np.ndarray): y-axis.
    """
    return m * x + n + a * exp(-((x - x0) ** 2) / (2 * sigma**2))


def get_fwhm_map_min_from_projection(
    lines: list, output_folder: str, label: str, pixel_step: int, plots_flag: bool
) -> tuple:
    """
    Open FWHM grid search optmization plot, then fit the projection in both axis to get the point of minimum FWHM of the azimuthal average.

    Args:
        lines (list): Output of grid search for FWHM optmization, each line must contain a values of xc,yc,fwhm,r_square.
        output_folder (str): Path to the folder where plots are saved.
        label (str): Plots filename label.
        pixel_step (str): Step size between grid points in pixels.
        plots_flag (bool): If True, plots can be generated.

    Returns:
        center (list): Coordinates of the center of the diffraction pattern in x and y.
    """
    n = int(math.sqrt(len(lines)))

    merged_dict = {}
    for dictionary in lines[:]:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    # Extract x, y, and z from merged_dict

    x_grid = np.array(merged_dict["xc"], dtype=np.int16).reshape((n, n))
    y_grid = np.array(merged_dict["yc"], dtype=np.int16).reshape((n, n))
    z_grid = np.array(merged_dict["fwhm"], dtype=np.float32).reshape((n, n))
    r_grid = np.array(merged_dict["r_square"], dtype=np.float32).reshape((n, n))
    z_grid = np.nan_to_num(z_grid)
    r_grid = np.nan_to_num(r_grid)

    proj_x = np.mean(z_grid, axis=1)
    proj_y = np.mean(z_grid, axis=0)

    x_vals = np.arange(np.min(x_grid), np.max(x_grid) + pixel_step, pixel_step)
    y_vals = np.arange(np.min(y_grid), np.max(y_grid) + pixel_step, pixel_step)

    try:
        xc = x_vals[np.argmin(proj_x)]
        yc = y_vals[np.argmin(proj_y)]
    except (ValueError, IndexError):
        xc = -1.0
        yc = -1.0

    if plots_flag:
        # Create a figure with three subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        pos1 = ax1.imshow(z_grid, cmap="rainbow")

        n = z_grid.shape[0]
        step = 2

        ax1.set_xticks(np.arange(0, n, step, dtype=int))
        ax1.set_yticks(np.arange(0, n, step, dtype=int))
        ax1.set_xticklabels(
            np.arange(x_vals[0], x_vals[-1] + 1, step, dtype=int), rotation=45
        )
        ax1.set_yticklabels(np.arange(y_vals[0], y_vals[-1] + 1, step, dtype=int))

        ax1.set_ylabel("yc [px]")
        ax1.set_xlabel("xc [px]")
        ax1.set_title("FWHM")

        pos2 = ax2.imshow(r_grid, cmap="rainbow")
        ax2.set_xticks(np.arange(0, n, step, dtype=int))
        ax2.set_yticks(np.arange(0, n, step, dtype=int))
        ax2.set_xticklabels(
            np.arange(x_vals[0], x_vals[-1] + 1, step, dtype=int), rotation=45
        )
        ax2.set_yticklabels(np.arange(y_vals[0], y_vals[-1] + 1, step, dtype=int))

        ax2.set_ylabel("yc [px]")
        ax2.set_xlabel("xc [px]")
        ax2.set_title("R²")

        ax3.scatter(x_vals, proj_x, color="b")
        ax3.scatter(xc, proj_x[np.argmin(proj_x)], color="r", label=f"xc: {xc}")
        ax3.set_ylabel("Average FWHM")
        ax3.set_xlabel("xc [px]")
        ax3.set_title("FWHM projection in x")
        ax3.legend()
        ax4.scatter(y_vals, proj_y, color="b")
        ax4.scatter(yc, proj_y[np.argmin(proj_y)], color="r", label=f"yc: {yc}")
        ax4.set_ylabel("Average FWHM")
        ax4.set_xlabel("yc [px]")
        ax4.set_title("FWHM projection in y")
        ax4.legend()
        fig.colorbar(pos1, ax=ax1, shrink=0.6)
        fig.colorbar(pos2, ax=ax2, shrink=0.6)

        path = pathlib.Path(f"{output_folder}/fwhm_map/")
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_folder}/fwhm_map/{label}.png")
        plt.close()

    if int(np.sum(proj_y)) == 0 or int(np.sum(proj_x)) == 0:
        xc = -1
        yc = -1

    return [np.round(xc, 0), np.round(yc, 0)]


def circle_mask(data: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    """
    Make a circular mask for the data.

    Args:
        data (np.ndarray): Image in which mask will be shaped.
        radius (int): Outer radius of the mask, in pixels.

    Returns:
        mask (np.ndarray): Mask array containg zeros (pixels to be masked) and ones (valid pixels).
    """

    a = data.shape[0]
    b = data.shape[1]

    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    return (np.greater(R, radius)).astype(np.int32)


def ring_mask(
    data: np.ndarray, center: tuple, inner_radius: int, outer_radius: int
) -> np.ndarray:
    """
    Make a ring mask for the data.

    Args:
        data (np.ndarray): Image in which mask will be shaped.
        center (tuple): Center coordinates (xc,yc) of the concentric rings for the mask.
        inner_radius (int): Inner radius of the mask, in pixels.
        outer_radius (int): Outer radius of the mask, in pixels.

    Returns:
        mask (np.ndarray): Mask array containg zeros (pixels to be masked) and ones (valid pixels).
    """

    bin_size = bin
    a = data.shape[0]
    b = data.shape[1]
    [X, Y] = np.meshgrid(np.arange(b) - center[0], np.arange(a) - center[1])
    R = np.sqrt(np.square(X) + np.square(Y))
    bin_size = outer_radius - inner_radius
    return np.greater(R, outer_radius - bin_size) & np.less(R, outer_radius + bin_size)


def visualize_single_panel(
    data: np.ndarray, transformation_matrix: np.ndarray, ss_in_rows: bool
) -> np.ndarray:
    """
    Creates a visulization array for single panel detectors after applying the detector geometry.

    Args:
        data (np.ndarray): Image in which mask will be shaped
        transformation_matrix (np.ndarray): A 2x2 transformation matrix used to map from fast-scan/slow-scan to x/y coordinates.
        ss_in_rows (bool): If True, the slow-scan axis is mapped to rows; otherwise to columns.

    Returns:
        np.ndarray: The transformed visualization array.
    """
    visual_data = np.full((2 * max(data.shape) + 1, 2 * max(data.shape) + 1), np.nan)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            point = (i, j) if ss_in_rows else (j, i)
            xy_j, xy_i = fsss_to_xy(point, transformation_matrix)
            visual_data[xy_i][xy_j] = data[i][j]

    non_nan_indices = np.where(~np.isnan(visual_data))
    min_row_index, min_col_index = np.min(non_nan_indices, axis=1)
    max_row_index, max_col_index = np.max(non_nan_indices, axis=1)

    return visual_data[
        min_row_index : max_row_index + 1, min_col_index : max_col_index + 1
    ]


def fsss_to_xy(point: tuple, m: list) -> tuple:
    """
    Transforms from the fast-scan/slow-scan basis to the x/y basis.

    Args:
        point (tuple): Coordinates in the fast-scan/slow-scan basis (ss,fs).
        m (list): A 2x2 transformation matrix.

    Returns:
        tuple: The corresponding (x, y) coordinates.
    """

    d = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    ss = point[0] + 1
    fs = point[1] + 1
    x = int((m[1][1] / d) * fs - (m[0][1] / d) * ss)
    y = int(-(m[1][0] / d) * fs + (m[0][0] / d) * ss)
    return x, y
