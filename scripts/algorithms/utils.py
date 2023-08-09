from typing import List, Optional, Callable, Tuple, Any, Dict
from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pandas as pd

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


def shift_image_by_n_pixels(data: np.ndarray, n: float, axis: int) -> np.ndarray:
    max_row, max_col = data.shape
    # print(max_row,max_col)
    if axis == 1 and n >= 0:
        shifted_image = np.pad(data, pad_width=[(0, 0), (abs(n), 0)], mode="constant")
        image_cut = shifted_image[:max_row, :max_col]
    elif axis == 1 and n < 0:
        shifted_image = np.pad(data, pad_width=[(0, 0), (0, abs(n))], mode="constant")
        image_cut = shifted_image[:max_row, abs(n) :]
    elif axis == 0 and n >= 0:
        shifted_image = np.pad(data, pad_width=[(abs(n), 0), (0, 0)], mode="constant")
        image_cut = shifted_image[:max_row, :max_col]
    elif axis == 0 and n < 0:
        shifted_image = np.pad(data, pad_width=[(0, abs(n)), (0, 0)], mode="constant")
        image_cut = shifted_image[abs(n) :, :max_col]
    # print("Image cut shape", image_cut.shape)
    return image_cut

def open_cc_map(lines: list, label: str = None):
    """
    Open autocorrelation grid search optmization plot, fit projections in both axis to get the point of maximum autocorrelation of Bragg peaks positions.
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
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Extract x, y, and z from merged_dict

    x = np.array(merged_dict["xc"]).reshape((n, n))[0]
    y = np.array(merged_dict["yc"]).reshape((n, n))[:, 0]
    z1 = np.nan_to_num(np.array(merged_dict["cc_0"]).reshape((n, n)))
    z2 = np.nan_to_num(np.array(merged_dict["cc_1"]).reshape((n, n)))


    
    #norm_z1=z1**2/np.max(z1**2)
    #norm_z1=z1/np.max(z1)
    #norm_z1=z1
    index_y, index_x = np.where(z1==np.max(z1))
    print('xc yc',x[index_x], y[index_y])
    pos1 = ax1.imshow(z1, cmap="nipy_spectral")
    step = 10
    n = z1.shape[0]
    ax1.set_xticks(np.arange(0, n, step, dtype=int))
    ax1.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax1.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax1.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))
    ax1.set_ylabel("yc [px]")
    ax1.set_xlabel("xc [px]")
    ax1.set_title("cc matrix sum")

    index_y, index_x = np.where(z2==np.max(z2))
    print('xc yc',x[index_x], y[index_y])
    pos2 = ax2.imshow(z2, cmap="nipy_spectral")
    step = 10
    n = z2.shape[0]
    ax2.set_xticks(np.arange(0, n, step, dtype=int))
    ax2.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax2.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax2.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))
    ax2.set_ylabel("yc [px]")
    ax2.set_xlabel("xc [px]")
    ax2.set_title("cc matrix max")
    """
    z=np.multiply(norm_z1,norm_z2)
    pos3 = ax3.imshow(z, cmap="nipy_spectral")
    step = 10
    n = z.shape[0]
    ax3.set_xticks(np.arange(0, n, step, dtype=int))
    ax3.set_yticks(np.arange(0, n, step, dtype=int))
    step = step * (abs(x[0] - x[1]))
    ax3.set_xticklabels(np.arange(x[0], x[-1] + step, step, dtype=int))
    ax3.set_yticklabels(np.arange(y[0], y[-1] + step, step, dtype=int))
    ax3.set_ylabel("yc [px]")
    ax3.set_xlabel("xc [px]")
    ax3.set_title("cc both")
    """

    fig.colorbar(pos1, ax=ax1, shrink=0.6)
    fig.colorbar(pos2, ax=ax2, shrink=0.6)
    #fig.colorbar(pos3, ax=ax3, shrink=0.6)

    # Display the figure

    # plt.show()
    plt.savefig(f'/home/rodria/Desktop/cc_map/lyso_{label}.png')
    # plt.close()


def table_of_center(
    crystal: int, rot: int, center_file: str = None, loaded_table_center: Dict = None
) -> List[int]:

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
    center_theory = []

    for i in files_path:

        label = str(i).split("/")[-1]
        # print(label)
        # crystal = int(label.split("_")[0][-2:])
        # rot = int(label.split("_")[1][-3:])
        crystal = int(label.split("_")[-3][-2:])
        rot = int(label.split("_")[-2][:])
        # print(crystal, rot)
        center, loaded_table_center = table_of_center(
            crystal, rot, center_file, loaded_table_center
        )
        center_theory.append(center)
    # print(center_theory)
    center_theory = np.array(center_theory)
    return center_theory, loaded_table_center


def extend_image(img:np.ndarray)->np.ndarray:
    row, col = img.shape
    print(row, col)
    extended_img=np.pad(img, pad_width=((0,0),(col,col)))
    extended_img=np.pad(extended_img, pad_width=((row,row),(0,0)))
    extended_img[np.where(extended_img)==0]=np.nan
    return extended_img

def shift_and_calculate_cc(shift: tuple)-> Dict[str, float]:
    """
    Wrong
    """
    #print(shift)
    shift_x = -shift[0]
    shift_y = -shift[1]
    xc = round(data.shape[1]/2) + shift[0]
    yc = round(data.shape[0]/2) + shift[1]
    #print(xc,yc)
    shifted_data=shift_image_by_n_pixels(shift_image_by_n_pixels(data, shift_y, 0), shift_x, 1)
    shifted_mask=shift_image_by_n_pixels(shift_image_by_n_pixels(xds_mask, shift_y, 0), shift_x, 1)
    pf8_info = PF8Info(
        max_num_peaks=10000,
        pf8_detector_info=dict(
            asic_nx=shifted_mask.shape[1],
            asic_ny=shifted_mask.shape[0],
            nasics_x=1,
            nasics_y=1,
        ),
        adc_threshold=10,
        minimum_snr=5,
        min_pixel_count=1,
        max_pixel_count=200,
        local_bg_radius=3,
        min_res=0,
        max_res=100,
        _bad_pixel_map=shifted_mask,
    )

    pf8 = PF8(pf8_info)

    peak_list = pf8.get_peaks_pf8(data=shifted_data)
    
    flipped_data=shifted_data[::-1,::-1]
    pf8_info._bad_pixel_map=shifted_mask[::-1,::-1]
    pf8 = PF8(pf8_info)

    peak_list_flipped = pf8.get_peaks_pf8(data=flipped_data)
    
    if peak_list["num_peaks"]>=peak_list_flipped["num_peaks"]:
        n_peaks=peak_list_flipped["num_peaks"]
        indices = (
        np.array(peak_list["ss"][:n_peaks], dtype=int),
        np.array(peak_list["fs"][:n_peaks], dtype=int),
        )    
        indices_flipped = (
        np.array(peak_list_flipped["ss"], dtype=int),
        np.array(peak_list_flipped["fs"], dtype=int),
        )
    else:
        n_peaks=peak_list["num_peaks"]
        indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
        )    
        indices_flipped = (
        np.array(peak_list_flipped["ss"][:n_peaks], dtype=int),
        np.array(peak_list_flipped["fs"][:n_peaks], dtype=int),
        )
    """
    if peak_list["num_peaks"]>=peak_list_flipped["num_peaks"]:
        n_pads=peak_list["num_peaks"]-peak_list_flipped["num_peaks"]
        n_peaks=peak_list["num_peaks"]
        indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
        )    
        indices_flipped = (
        np.pad(np.array(peak_list_flipped["ss"], dtype=int), pad_width=(0, n_pads)),
        np.pad(np.array(peak_list_flipped["fs"], dtype=int), pad_width=(0, n_pads))
        )
    else:
        n_peaks=peak_list_flipped["num_peaks"]
        n_pads=peak_list_flipped["num_peaks"]-peak_list["num_peaks"]
        indices = (
        np.pad(np.array(peak_list["ss"], dtype=int),pad_width=(0, n_pads)),
        np.pad(np.array(peak_list["fs"], dtype=int),pad_width=(0, n_pads))
        )    
        indices_flipped = (
        np.array(peak_list_flipped["ss"], dtype=int),
        np.array(peak_list_flipped["fs"], dtype=int),
        )
        """
    #print(n_peaks)
    #print(peak_list['ss'], peak_list_flipped['ss'])
    #print(peak_list['fs'], peak_list_flipped['fs'])

    #fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2,figsize=(10, 10))
    #ax1.imshow(shifted_data,cmap='cividis',vmax=10)
    #ax1.scatter(data.shape[1]/2, data.shape[0]/2)
    #ax1.scatter(indices[1], indices[0], marker="o", edgecolor="r", facecolor="none", s=30)
    #ax2.imshow(flipped_data,cmap='cividis',vmax=10)
    #ax2.scatter(indices_flipped[1], indices_flipped[0], marker="o", edgecolor="lime", facecolor="none", s=30)
    #ax2.scatter(data.shape[1]/2, data.shape[0]/2)
    #ax3.scatter(np.sort(indices[1]), np.sort(indices_flipped[1]))

    peaks={'peaks_x': indices[1],
          'peaks_y': indices[0],
        }
    peaks_flipped={'peaks_x': indices_flipped[1],
          'peaks_y': indices_flipped[0]
        }

    df_orig=pd.DataFrame(peaks)
    df_flip=pd.DataFrame(peaks_flipped)
    

    cc_matrix = df_orig.corrwith(df_flip, axis=0)
    #print(cc_matrix)
    
    #print('norm cc x',cc_matrix.peaks_x)
    #print('norm cc y',cc_matrix.peaks_y)
    #print('sum', cc_matrix.peaks_x+cc_matrix.peaks_y)
    #ax4.scatter(indices[0], indices_flipped[0])
    #plt.show()
    return {
        "shift_x": shift_x,
        "shift_y": shift_y,
        "xc": xc,
        "yc": yc,
        "cc_0": cc_matrix.peaks_x,
        "cc_1": cc_matrix.peaks_y
    }

def shift_and_calculate_autocorrelation(shift: tuple)-> Dict[str, float]:
    """
    Wrong
    """
    
    pf8_info = PF8Info(
        max_num_peaks=10000,
        pf8_detector_info=dict(
            asic_nx=xds_mask.shape[1],
            asic_ny=xds_mask.shape[0],
            nasics_x=1,
            nasics_y=1,
        ),
        adc_threshold=10,
        minimum_snr=5,
        min_pixel_count=1,
        max_pixel_count=200,
        local_bg_radius=3,
        min_res=0,
        max_res=85,
        _bad_pixel_map=xds_mask,
    )

    pf8 = PF8(pf8_info)

    peak_list = pf8.get_peaks_pf8(data=data)
    
    shift_x = shift[0]
    shift_y = shift[1]
    xc = data.shape[1]/2 + shift[0]/2
    yc = data.shape[0]/2 + shift[1]/2
    flipped_data=data[::-1,::-1]
    flipped_mask=xds_mask[::-1,::-1]
    shifted_data=shift_image_by_n_pixels(shift_image_by_n_pixels(flipped_data, shift_y, 0), shift_x, 1)
    shifted_mask=shift_image_by_n_pixels(shift_image_by_n_pixels(flipped_mask, shift_y, 0), shift_x, 1)

    
    pf8_info._bad_pixel_map=shifted_mask
    pf8 = PF8(pf8_info)

    peak_list_flipped = pf8.get_peaks_pf8(data=shifted_data)
    

    if peak_list["num_peaks"]>=peak_list_flipped["num_peaks"]:
        n_peaks=peak_list_flipped["num_peaks"]
        indices = (
        np.array(peak_list["ss"][:n_peaks], dtype=int),
        np.array(peak_list["fs"][:n_peaks], dtype=int),
        )    
        indices_flipped = (
        np.array(peak_list_flipped["ss"], dtype=int),
        np.array(peak_list_flipped["fs"], dtype=int),
        )
    else:
        n_peaks=peak_list["num_peaks"]
        indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
        )    
        indices_flipped = (
        np.array(peak_list_flipped["ss"][:n_peaks], dtype=int),
        np.array(peak_list_flipped["fs"][:n_peaks], dtype=int),
        )
    #print(n_peaks)
    #print(peak_list, peak_list_flipped)
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 5))
    #ax1.scatter(indices[1], indices[0])
    #ax1.scatter(indices_flipped[1], indices_flipped[0])

    #ax2.scatter(indices[1], indices_flipped[1])
    cc_0 = np.corrcoef(x=(indices[0], indices_flipped[0]))
    

    cc_1 = np.corrcoef(x=(indices[1], indices_flipped[1]))
    
    return {
        "shift_x": shift[0],
        "shift_y": shift[1],
        "xc": xc,
        "yc": yc,
        "cc_0": cc_0[0,1],
        "cc_1": cc_1[0,1]
    }
