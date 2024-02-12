from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import os
from find_center_friedel import calculate_center_friedel_pairs
import om.lib.geometry as geometry
import argparse
import numpy as np
from utils import (
    mask_peaks,
    gaussian,
    gaussian_lin,
    center_of_mass,
    azimuthal_average,
    open_fwhm_map_global_min,
    open_r_sqrd_map_global_max,
    circle_mask,
    update_corner_in_geom,
    correct_polarization_python,
    ring_mask,
)
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from models import PF8, PF8Info
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt

plt.switch_backend("agg")
from scipy import signal
import subprocess as sub
import h5py
import hdf5plugin
import multiprocessing
import settings

config=settings.read("config.yaml")

BeambustersParam=settings.parse(config)

PF8Config = PF8Info(
    max_num_peaks=config["pf8"]["max_num_peaks"],
    adc_threshold=config["pf8"]["adc_threshold"],
    minimum_snr=config["pf8"]["minimum_snr"],
    min_pixel_count=config["pf8"]["min_pixel_count"],
    max_pixel_count=config["pf8"]["max_pixel_count"],
    local_bg_radius=config["pf8"]["local_bg_radius"],
    min_res=config["pf8"]["min_res"],
    max_res=config["pf8"]["max_res"],
)

def calculate_fwhm(data_and_coordinates: tuple) -> Dict[str, int]:
    corrected_data, mask, center_to_radial_average, plots_info = data_and_coordinates

    x, y = azimuthal_average(corrected_data, center=center_to_radial_average, mask=mask)
    x_all = x.copy()
    y_all = y.copy()

    if plots_info["plot_flag"]:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(x, y)

    ## Define background peak region
    x_min = config["peak_region"]["min"]
    x_max = config["peak_region"]["max"]
    x = x[x_min:x_max]
    y = y[x_min:x_max]
    ## Estimation of initial parameters

    m0 = 0
    n0 = 2
    y_linear = m0 * x + n0
    y_gaussian = y - y_linear

    mean = sum(x * y_gaussian) / sum(y_gaussian)
    sigma = np.sqrt(sum(y_gaussian * (x - mean) ** 2) / sum(y_gaussian))
    try:
        popt, pcov = curve_fit(
            gaussian_lin, x, y, p0=[max(y_gaussian), mean, sigma, m0, n0]
        )
        fwhm = popt[2] * math.sqrt(8 * np.log(2))
        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]

        ##Calculate residues
        residuals = y - gaussian_lin(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    except:
        r_squared = 0
        fwhm = 1000
        fwhm_over_radius = 1000
        popt = []

    ## Display plots
    if plots_info["plot_flag"] and len(popt) > 0:
        x_fit = x.copy()
        y_fit = gaussian_lin(x_fit, *popt)

        plt.vlines([x[0], x[-1]], 0, round(popt[0]) + 2, "r")

        plt.plot(
            x_fit,
            y_fit,
            "r--",
            label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
        )

        plt.title("Azimuthal integration")
        plt.xlim(0, 500)
        # plt.ylim(0, round(popt[0])+2)
        plt.legend()
        plt.savefig(
            f'{plots_info["args"].scratch}/center_refinement/plots/{plots_info["run_label"]}/radial_average/{plots_info["file_label"]}_{plots_info["frame_index"]}.png'
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
        description="Calculate center of diffraction patterns fro MHz beam sweeping serial crystallography using the beambusters software."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )
    parser.add_argument(
        "-g",
        "--geom",
        type=str,
        action="store",
        help="CrystFEL geometry filename",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        action="store",
        help="path to output folder for processed data",
    )
    parser.add_argument(
        "-s",
        "--scratch",
        type=str,
        action="store",
        help="path to output folder for scratch_cc where I save plots of each processed image ",
    )

    args = parser.parse_args()
    plot_flag = False

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    file_format = os.path.basename(args.input).split(".")[-1]

    ### Extract geometry file

    geometry_txt = open(args.geom, "r").readlines()
    mask_file = [
        x.split(" = ")[-1][:-1]
        for x in geometry_txt
        if x.split(" = ")[0] == "mask_file"
    ][0]
    mask_path = [
        x.split(" = ")[-1][:-1] for x in geometry_txt if x.split(" = ")[0] == "mask"
    ][0]
    clen = float(
        [
            (x.split(" = ")[-1]).split("; ")[0]
            for x in geometry_txt
            if x.split(" = ")[0] == "clen"
        ][0]
    )
    pixel_resolution = float(
        [(x.split(" = ")[-1]) for x in geometry_txt if x.split(" = ")[0] == "res"][0]
    )
    h5_path = [
        x.split(" = ")[-1][:-1] for x in geometry_txt if x.split(" = ")[0] == "data"
    ][0]
    geom = geometry.GeometryInformation(
        geometry_description=geometry_txt, geometry_format="crystfel"
    )
    pixel_maps = geom.get_pixel_maps()
    detector_layout = geom.get_layout_info()

    _img_center_x = -1 * pixel_maps["x"][0, 0]
    _img_center_y = -1 * pixel_maps["y"][0, 0]
    DetectorCenter = [_img_center_x, _img_center_y]
    data_visualize = geometry.DataVisualizer(pixel_maps=pixel_maps)

    ## Geometry info to Peakfinder8
    PF8Config.pf8_detector_info = detector_layout
    PF8Config.bad_pixel_map_filename = mask_file
    PF8Config.bad_pixel_map_hdf5_path = mask_path
    PF8Config.pixel_maps = pixel_maps

    with h5py.File(f"{mask_file}", "r") as f:
        mask = np.array(f[f"{mask_path}"])

    try:
        job_index = int(args.input[-2:])
    except ValueError:
        job_index = 0

    if file_format == "lst":
        for i in range(0, len(paths[:])):
            # Create run folder in processed folder
            file_name = str(paths[i][:-1])
            split_path = (os.path.dirname(file_name)).split("/")
            run_label = split_path[-2] + "/" + split_path[-1]
            file_label = os.path.basename(file_name).split(".")[0]
            if not os.path.exists(f"{args.output}/centered/{run_label}"):
                cmd = f"mkdir {args.output}/centered/{split_path[-2]}; mkdir {args.output}/centered/{run_label}"
                sub.call(cmd, shell=True)
            if os.path.exists(os.path.dirname(file_name)+"/info.txt"):
                cmd = f"cp {os.path.dirname(file_name)+'/info.txt'} {args.output}/centered/{run_label}"
                sub.call(cmd, shell=True)
            if config["plots_flag"] and not os.path.exists(
                f"{args.scratch}/center_refinement/plots/{run_label}"
            ):
                cmd = f"mkdir {args.scratch}/center_refinement/plots/{split_path[-2]}; mkdir {args.scratch}/center_refinement/plots/{run_label};"
                sub.call(cmd, shell=True)
                cmd = f" mkdir {args.scratch}/center_refinement/plots/{run_label}/distance_map/; mkdir {args.scratch}/center_refinement/plots/{run_label}/peaks/; mkdir {args.scratch}/center_refinement/plots/{run_label}/centered_friedel/; mkdir {args.scratch}/center_refinement/plots/{run_label}/radial_average/; mkdir {args.scratch}/center_refinement/plots/{run_label}/centered/;  mkdir {args.scratch}/center_refinement/plots/{run_label}/fwhm_map/;  mkdir {args.scratch}/center_refinement/plots/{run_label}/edges/"
                sub.call(cmd, shell=True)

            ## Open dataset
            data_file_format = os.path.basename(file_name).split(".")[-1]
            if data_file_format == "h5":
                f = h5py.File(f"{file_name}", "r")
                data = f[f"{h5_path}"]

            if not config["plots_flag"]:
                max_frame = data.shape[0]
            else:
                max_frame = 10
            _data_shape = data.shape
            ## Initialize collecting arrays
            raw_data = np.ndarray((_data_shape), dtype=np.int32)
            if not config["skip_pol"]:
                pol_correct_data = np.ndarray((_data_shape), dtype=np.int32)
            center_data = np.ndarray((_data_shape[0], 2), dtype=np.float32)
            center_data_method_zero = np.ndarray((_data_shape[0], 2), dtype=np.float32)
            center_data_method_one = np.ndarray((_data_shape[0], 2), dtype=np.float32)
            center_data_method_two = np.ndarray((_data_shape[0], 2), dtype=np.float32)
            center_data_method_three = np.ndarray((_data_shape[0], 2), dtype=np.float32)
            shift_x_mm = np.ndarray((_data_shape[0],), dtype=np.float32)
            shift_y_mm = np.ndarray((_data_shape[0],), dtype=np.float32)

            for frame_index in range(max_frame):
                ## For plots info path
                plots_info = {
                    "plot_flag": plot_flag,
                    "file_label": file_label,
                    "run_label": run_label,
                    "frame_index": frame_index,
                    "args": args,
                }

                frame = np.array(data[frame_index])
                raw_data[frame_index, :, :] = frame

                corrected_data = data_visualize.visualize_data(data=frame * mask)
                visual_mask = data_visualize.visualize_data(data=mask).astype(int)
                visual_mask[np.where(corrected_data < 0)] = 0
                ## Peakfinder8 detector information and bad_pixel_map
                ## Performing peak search

                pf8 = PF8(PF8Config)
                peak_list = pf8.get_peaks_pf8(data=frame)
                peak_list_x_in_frame, peak_list_y_in_frame = pf8.peak_list_in_slab(
                    peak_list
                )
                indices = np.ndarray((2, peak_list["num_peaks"]), dtype=int)

                for idx, k in enumerate(peak_list_y_in_frame):
                    row_peak = int(k + DetectorCenter[1])
                    col_peak = int(peak_list_x_in_frame[idx] + DetectorCenter[0])
                    indices[0, idx] = row_peak
                    indices[1, idx] = col_peak

                if 0 not in config["skip_method"]:
                    # Mask Bragg  peaks
                    if config["bragg_pos_center_of_mass"] == 1:
                        only_peaks_mask = mask_peaks(
                            visual_mask, indices, bragg=1, n=config["pixels_per_peak"]
                        )
                    elif config["bragg_pos_center_of_mass"] == 0:
                        only_peaks_mask = mask_peaks(
                            visual_mask, indices, bragg=0, n=config["pixels_per_peak"]
                        )
                    elif config["bragg_pos_center_of_mass"] == -1:
                        only_peaks_mask = mask_peaks(
                            visual_mask, indices, bragg=-1, n=config["pixels_per_peak"]
                        )
                    first_mask = only_peaks_mask * visual_mask

                    ## TEST set intesities to one
                    #unity_data= corrected_data.copy()
                    #unity_data[first_mask]=1
                    #converged,center_from_method_zero = center_of_mass(unity_data, first_mask)

                    converged, center_from_method_zero = center_of_mass(
                        corrected_data, first_mask
                    )
                    if converged == 0:
                        center_from_method_zero = DetectorCenter.copy()
                    center_from_method_zero[0] += config["offset"]["x"]
                    center_from_method_zero[1] += config["offset"]["y"]
                else:
                    center_from_method_zero = [0, 0]

                center_data_method_zero[frame_index, :] = center_from_method_zero
                # Mask Bragg peaks. I take the mask shape, Bragg peaks positions, bragg is a flag if I want to see only bragg peaks or only the image without Bragg peaks

                if 1 not in config["skip_method"]:
                    only_peaks_mask = mask_peaks(
                        visual_mask, indices, bragg=0, n=config["pixels_per_peak"]
                    )
                    pf8_mask = only_peaks_mask * visual_mask
                    ## Scikit-image circle detection
                    edges = canny(
                        corrected_data,
                        mask=pf8_mask,
                        sigma=config["canny"]["sigma"],
                        use_quantiles=True,
                        low_threshold=config["canny"]["low_threshold"],
                        high_threshold=config["canny"]["high_threshold"],
                    )
                    if config["plots_flag"]:
                        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
                        ax1.imshow(edges)
                        plt.savefig(
                            f"{args.scratch}/center_refinement/plots/{run_label}/edges/{file_label}_{frame_index}.png"
                        )
                        plt.close()
                    # Detect radii
                    hough_radii = np.arange(config["peak_region"]["min"], config["peak_region"]["max"], 1)
                    hough_res = hough_circle(edges, hough_radii)
                    # Select the most prominent 1 circle
                    accums, cx, cy, radii = hough_circle_peaks(
                        hough_res, hough_radii, total_num_peaks=1
                    )
                    if len(cx) > 0:
                        center_from_method_one = [cx[0], cy[0]]
                    else:
                        center_from_method_one = center_from_method_zero.copy()
                else:
                    center_from_method_one = [0, 0]
                center_data_method_one[frame_index, :] = center_from_method_one
                ## Calculate FWHM of the background peak for each coordinate in a box of OutlierDistance around the pixel coordinates defined in center_from_method_zero
                if 2 not in config["skip_method"]:
                    pixel_step = 1
                    xx, yy = np.meshgrid(
                        np.arange(
                            center_from_method_one[0] - config["outlier_distance"],
                            center_from_method_one[0] + config["outlier_distance"] + 1,
                            pixel_step,
                            dtype=int,
                        ),
                        np.arange(
                            center_from_method_one[1] - config["outlier_distance"],
                            center_from_method_one[1] + config["outlier_distance"] + 1,
                            pixel_step,
                            dtype=int,
                        ),
                    )
                    coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                    masked_data = corrected_data.copy()

                    ## TEST avoid effects from secondary peaks
                    # ring_mask_array=ring_mask(masked_data,center_from_method_one, config["peak_region"]["min"], config["peak_region"]["max"])
                    # masked_data[~ring_mask_array]=0

                    coordinates_anchor_data = [
                        (masked_data, pf8_mask, shift, plots_info)
                        for shift in coordinates
                    ]
                    fwhm_summary = []
                    pool = multiprocessing.Pool()
                    with pool:
                        fwhm_summary = pool.map(calculate_fwhm, coordinates_anchor_data)

                    center_from_method_two = open_fwhm_map_global_min(
                        fwhm_summary,
                        f"{args.scratch}/center_refinement/plots/{run_label}/fwhm_map/{file_label}_{frame_index}",
                        pixel_step,
                        config["plots_flag"],
                    )
                else:
                    center_from_method_two = [0, 0]
                center_data_method_two[frame_index, :] = center_from_method_two

                if config["method"] == 0:
                    xc, yc = center_from_method_zero
                elif config["method"] == 1:
                    xc, yc = center_from_method_one
                elif config["method"] == 2:
                    xc, yc = center_from_method_two

                if config["force_center"]["mode"]:
                    xc=config["force_center"]["x"]
                    yc=config["force_center"]["y"]

                if peak_list["num_peaks"] >= 4:
                    PF8Config.modify_radius(
                        -xc + DetectorCenter[0], -yc + DetectorCenter[1]
                    )
                    pf8 = PF8(PF8Config)
                    peak_list = pf8.get_peaks_pf8(data=frame)
                    peak_list_in_slab = pf8.peak_list_in_slab(peak_list)
                    center_from_method_three = calculate_center_friedel_pairs(
                        corrected_data,
                        visual_mask,
                        peak_list_in_slab,
                        [xc, yc],
                        config["search_radius"],
                        config["outlier_distance"],
                        config["plots_flag"],
                        f"{args.scratch}/center_refinement/plots/{run_label}",
                        f"{file_label}_{frame_index}",
                    )
                else:
                    center_from_method_three = None
                if center_from_method_three:
                    center_data_method_three[frame_index, :] = center_from_method_three
                    xc, yc = center_from_method_three
                else:
                    center_data_method_three[frame_index, :] = [-1, -1]
                    
                    if config["force_center"]["mode"]:
                        xc = config["force_center"]["x"]
                        yc = config["force_center"]["y"]
                    else:
                        if config["method"] == 0:
                            xc, yc = center_from_method_zero
                        elif config["method"] == 1:
                            xc, yc = center_from_method_one
                        elif config["method"] == 2:
                            xc, yc = center_from_method_two

                ## Here you get the direct beam position in detector coordinates.
                refined_center = (np.round(xc, 4), np.round(yc, 4))
                center_data[frame_index, :] = refined_center

                detector_shift_x = (
                    refined_center[0] - DetectorCenter[0]
                ) * 1e3 / pixel_resolution
                detector_shift_y = (
                    refined_center[1] - DetectorCenter[1]
                )  * 1e3 / pixel_resolution

                shift_x_mm[frame_index] = np.round(detector_shift_x,4)
                shift_y_mm[frame_index] = np.round(detector_shift_y,4)
                ## For the calculated direct beam postion I do the azimuthal integration and save the plot to check results.

                if config["plots_flag"] and 2 not in config["skip_method"]:
                    plot_flag = True
                    results = calculate_fwhm(
                        (corrected_data, pf8_mask, (refined_center))
                    )
                    plot_flag = False

                ## Display plots to check peaksearch and if the refined center converged.
                if config["plots_flag"]:
                    xr = center_from_method_zero[0]
                    yr = center_from_method_zero[1]
                    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
                    pos1 = ax1.imshow(
                        corrected_data * visual_mask, vmax=200, cmap="YlGn"
                    )
                    ax1.scatter(
                        round(DetectorCenter[0]),
                        round(DetectorCenter[1]),
                        color="blue",
                        edgecolor="k",
                        s=50,
                        linewidth=1,
                        label=f"Initial detector center:({round(DetectorCenter[0])},{round(DetectorCenter[1])})",
                    )
                    if 0 not in config["skip_method"]:
                        ax1.scatter(
                            round(center_from_method_zero[0]),
                            round(center_from_method_zero[1]),
                            color="magenta",
                            marker="+",
                            s=150,
                            label=f"First center:({round(center_from_method_zero[0])},{round(center_from_method_zero[1])})",
                        )
                    if 1 not in config["skip_method"]:
                        ax1.scatter(
                            round(center_from_method_one[0]),
                            round(center_from_method_one[1]),
                            color="magenta",
                            marker="+",
                            s=150,
                            label=f"Second center:({round(center_from_method_one[0])},{round(center_from_method_one[1])})",
                        )
                    if 2 not in config["skip_method"]:
                        ax1.scatter(
                            round(center_from_method_two[0]),
                            round(center_from_method_two[1]),
                            color="green",
                            marker="+",
                            s=150,
                            label=f"Third center:({round(center_from_method_two[0])}, {round(center_from_method_two[1])})",
                        )
                    ax1.scatter(
                        indices[1],
                        indices[0],
                        facecolor="none",
                        s=100,
                        marker="o",
                        edgecolor="orangered",
                        label="Bragg peaks",
                    )
                    ax1.scatter(
                        round(refined_center[0]),
                        round(refined_center[1]),
                        color="red",
                        edgecolor="k",
                        marker="o",
                        s=50,
                        linewidth=0.5,
                        label=f"Refined detector center:({round(refined_center[0])},{round(refined_center[1])})",
                    )
                    ax1.set_title("Detector center refinement")
                    fig.colorbar(pos1, ax=ax1, shrink=0.6)
                    ax1.set_xlim(200, 900)
                    ax1.set_ylim(900, 200)
                    ax1.legend(fontsize="10", loc="upper left")
                    plt.savefig(
                        f"{args.scratch}/center_refinement/plots/{run_label}/centered/{file_label}_{frame_index}.png"
                    )
                    plt.close()

                if not config["skip_pol"]:
                    PF8Config.modify_radius(
                        -xc + refined_center[0], -yc + refined_center[1]
                    )
                    pf8 = PF8(PF8Config)
                    pixel_maps = pf8.pf8_param.pixel_maps
                    pol_corrected_frame, pol_array_first = correct_polarization_python(
                        pixel_maps["x"],
                        pixel_maps["y"],
                        float(clen * pixel_resolution),
                        frame,
                        mask=mask,
                        p=0.5,
                    )
                    pol_correct_data[frame_index, :, :] = np.array(
                        pol_corrected_frame, dtype=np.int32
                    )

                ## Reset geom
                geometry_txt = open(args.geom, "r").readlines()
                geom = geometry.GeometryInformation(
                    geometry_description=geometry_txt, geometry_format="crystfel"
                )
                pixel_maps = geom.get_pixel_maps()
                PF8Config.pixel_maps = pixel_maps
            f.close()

            if config["plots_flag"]:
                output_folder = f"{args.scratch}/centered"
            else:
                output_folder = f"{args.output}/centered/{run_label}"

            with h5py.File(f"{output_folder}/{file_label}.h5", "w") as f:
                ## Here comes everything needed to pass to CrystFEL.
                entry = f.create_group("entry")
                entry.attrs["NX_class"] = "NXentry"
                grp_data = entry.create_group("data")
                grp_data.attrs["NX_class"] = "NXdata"
                grp_data.create_dataset("data", data=raw_data, compression="gzip")
                if not config["skip_pol"]:
                    grp_data.create_dataset(
                        "pol_corrected_data",
                        data=pol_corrected_data,
                        compression="gzip",
                    )
                f.create_dataset(
                    "shift_x_in_frame_mm", data=shift_x_mm, compression="gzip"
                )
                f.create_dataset(
                    "shift_y_in_frame_mm", data=shift_y_mm, compression="gzip"
                )
                grp_config = f.create_group("beambusters_config")
                grp_config.attrs["NX_class"] = "NXdata"
                for key, value in BeambustersParam.items():
                    grp_config.create_dataset(key, data=value)
                grp_config.create_dataset(
                    "refined_center", data=center_data, compression="gzip"
                )
                grp_config.create_dataset(
                    "center_from_method_zero",
                    data=center_data_method_zero,
                    compression="gzip",
                )
                grp_config.create_dataset(
                    "center_from_method_one",
                    data=center_data_method_one,
                    compression="gzip",
                )
                grp_config.create_dataset(
                    "center_from_method_two",
                    data=center_data_method_two,
                    compression="gzip",
                )
                grp_config.create_dataset(
                    "center_from_method_three",
                    data=center_data_method_three,
                    compression="gzip",
                )
                grp_config.create_dataset("geometry_file", data=args.geom)
                grp_config.create_dataset(
                    "detector_center", data=DetectorCenter, compression="gzip"
                )
                grp_config.create_dataset(
                    "pixel_resolution", data=pixel_resolution
                )
                grp_config.create_dataset(
                    "clen", data=clen
                )


if __name__ == "__main__":
    main()
