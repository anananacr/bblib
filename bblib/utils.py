from numpy import exp
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import math


def center_of_mass(data: np.ndarray, mask: np.ndarray = None) -> list[int]:
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
    intensity = np.reshape(data.copy(), len(mask))
    pol = mask.copy().astype(np.float32)
    pol = make_polarization_array(pol, x.flatten(), y.flatten(), dist, p)
    intensity = intensity / pol
    return intensity.reshape(data.shape), pol.reshape(data.shape)


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


def mask_peaks(mask: np.ndarray, indexes: tuple, bragg: int, n: int) -> np.ndarray:
    """
    Gather coordinates of a box of 1x1 pixels around each point from the indexes list. Bragg flag indicates if the mask returned will contain only bragg peaks regions (bragg =1), no bragg peaks regions (bragg=0), or both (bragg =-1).
    Parameters
    ----------
    mask: np.ndarray
        An array where mask will be built based on its shape. Mask shape is the same size of data.
    indexes: tuple
        Bragg peaks coordinates, indexes[0] contains x-coordinates of Bragg peaks and indexes[1] the corresponding y-coordinates.
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


def get_fwhm_map_global_min(
    lines: list, output_folder: str, label: str, pixel_step: int, plots_flag: bool
) -> tuple:
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
    pos1 = ax1.imshow(z, cmap="rainbow")
    step = 10
    n = z.shape[0]

    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))

    step = round(step * (abs(x[0] - x[1])), 1)
    ax1.set_xticklabels(
        np.arange(round(x[0], 1), round(x[-1] + step, 1), step, dtype=int), rotation=45
    )
    ax1.set_yticklabels(
        np.arange(round(y[0], 1), round(y[-1] + step, 1), step, dtype=int)
    )
    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("FWHM")

    pos2 = ax2.imshow(r, cmap="rainbow")
    step = 10
    n = z.shape[0]

    ax2.set_xticks(np.arange(0, n, step, dtype=int))
    ax2.set_yticks(np.arange(0, n, step, dtype=int))

    step = round(step * (abs(x[0] - x[1])), 1)
    ax2.set_xticklabels(
        np.arange(round(x[0], 1), round(x[-1] + step, 1), step, dtype=int), rotation=45
    )
    ax2.set_yticklabels(
        np.arange(round(y[0], 1), round(y[-1] + step, 1), step, dtype=int)
    )

    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("R²")

    proj_x = np.sum(z, axis=0) // n
    x = np.arange(x[0], x[-1] + pixel_step, pixel_step)
    index_x = np.unravel_index(np.argmin(proj_x, axis=None), proj_x.shape)
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

    if int(np.sum(proj_y)) == 0 or int(np.sum(proj_x)) == 0:
        xc = -1
        yc = -1
    else:
        if plots_flag:
            plt.savefig(f"{output_folder}/fwhm_map/{label}.png")
    plt.close()
    return [np.round(xc, 0), np.round(yc, 0)]


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


def ring_mask(
    data: np.ndarray, center: tuple, inner_radius: int, outer_radius: int
) -> np.ndarray:
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


def visualize_single_panel(
    data: np.ndarray, transformation_matrix: np.ndarray, ss_in_rows: bool
) -> np.ndarray:
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
    d = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    ss = point[0] + 1
    fs = point[1] + 1
    x = int((m[1][1] / d) * fs - (m[0][1] / d) * ss)
    y = int(-(m[1][0] / d) * fs + (m[0][0] / d) * ss)
    return x, y
