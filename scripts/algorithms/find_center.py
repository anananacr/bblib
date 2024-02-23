from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import os
from find_center_friedel import calculate_center_friedel_pairs
import om.lib.geometry as geometry
import argparse
import numpy as np
from utils import mask_peaks, correct_polarization
from models import PF8
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import subprocess as sub
import h5py
import hdf5plugin
import multiprocessing
import settings
from methods import CenterOfMass, CircleDetection, MinimizePeakFWHM


config = settings.read("config.yaml")
BeambustersParam = settings.parse(config)
PF8Config = settings.get_pf8_info(config)


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

    if detector_layout["nasics_x"] * detector_layout["nasics_y"] > 1:
        # Multiple panels
        # Get minimum array shape
        y_minimum = (
            2 * int(max(abs(pixel_maps["y"].max()), abs(pixel_maps["y"].min()))) + 2
        )
        x_minimum = (
            2 * int(max(abs(pixel_maps["x"].max()), abs(pixel_maps["x"].min()))) + 2
        )
        visual_img_shape = (y_minimum, x_minimum)
        # Detector center in the middle of the minimum array
        _img_center_x = int(visual_img_shape[1] / 2)
        _img_center_y = int(visual_img_shape[0] / 2)
    else:
        # Single panel
        _img_center_x = int(abs(pixel_maps["x"][0, 0]))
        _img_center_y = int(abs(pixel_maps["y"][0, 0]))

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
            if os.path.exists(os.path.dirname(file_name) + "/info.txt"):
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
                starting_frame = 0
            else:
                max_frame = 10
                starting_frame = config["starting_frame"]
            _data_shape = data.shape

            ## Initialize collecting arrays
            raw_data = np.ndarray(
                (max_frame, _data_shape[1], _data_shape[2]), dtype=np.int32
            )
            if not config["skip_pol"]:
                pol_correct_data = np.ndarray((_data_shape), dtype=np.int32)
            center_data = np.ndarray((max_frame, 2), dtype=np.float32)
            center_data_method_zero = np.ndarray((max_frame, 2), dtype=np.float32)
            center_data_method_one = np.ndarray((max_frame, 2), dtype=np.float32)
            center_data_method_two = np.ndarray((max_frame, 2), dtype=np.float32)
            center_data_method_three = np.ndarray((max_frame, 2), dtype=np.float32)
            shift_x_mm = np.ndarray((max_frame,), dtype=np.float32)
            shift_y_mm = np.ndarray((max_frame,), dtype=np.float32)

            for frame_index in range(max_frame):
                ## Path where plots will be saved
                plots_info = {
                    "plot_flag": plot_flag,
                    "file_label": file_label,
                    "run_label": run_label,
                    "frame_index": frame_index,
                    "args": args,
                }

                frame = np.array(data[starting_frame + frame_index])
                raw_data[frame_index, :, :] = frame

                corrected_data = data_visualize.visualize_data(data=frame * mask)
                visual_mask = data_visualize.visualize_data(data=mask).astype(int)
                visual_mask[np.where(corrected_data < 0)] = 0

                ## Precentering methods for first detector center approximation
                if 0 not in config["skip_method"]:
                    centering_method = CenterOfMass(config=config, PF8Config=PF8Config)
                    converged, center_from_method_zero = centering_method.__call__(
                        data=frame, initial_center=DetectorCenter
                    )
                else:
                    center_from_method_zero = [0, 0]

                center_data_method_zero[frame_index, :] = center_from_method_zero

                if 1 not in config["skip_method"]:
                    centering_method = CircleDetection(
                        config=config, PF8Config=PF8Config, plots_info=plots_info
                    )
                    converged, center_from_method_one = centering_method.__call__(
                        data=frame, initial_center=DetectorCenter
                    )
                else:
                    center_from_method_one = [0, 0]

                center_data_method_one[frame_index, :] = center_from_method_one

                if 2 not in config["skip_method"]:
                    centering_method = MinimizePeakFWHM(
                        config=config, PF8Config=PF8Config, plots_info=plots_info
                    )
                    converged, center_from_method_two = centering_method.__call__(
                        data=frame, initial_center=DetectorCenter
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
                    xc = config["force_center"]["x"]
                    yc = config["force_center"]["y"]

                ## Center refinement by Friedel pairs inversion symmetry
                PF8Config.modify_radius(
                    -xc + DetectorCenter[0], -yc + DetectorCenter[1]
                )
                pf8 = PF8(PF8Config)
                peak_list = pf8.get_peaks_pf8(data=frame)

                if peak_list["num_peaks"] >= 4:
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
                    (refined_center[0] - DetectorCenter[0]) * 1e3 / pixel_resolution
                )
                detector_shift_y = (
                    (refined_center[1] - DetectorCenter[1]) * 1e3 / pixel_resolution
                )

                shift_x_mm[frame_index] = np.round(detector_shift_x, 4)
                shift_y_mm[frame_index] = np.round(detector_shift_y, 4)

                ## Display plots to check peaksearch and if the center refinement looks good.
                if config["plots_flag"]:
                    peak_list_x_in_frame, peak_list_y_in_frame = peak_list_in_slab
                    indices = np.ndarray((2, peak_list["num_peaks"]), dtype=int)

                    for idx, k in enumerate(peak_list_y_in_frame):
                        row_peak = int(k + DetectorCenter[1])
                        col_peak = int(peak_list_x_in_frame[idx] + DetectorCenter[0])
                        indices[0, idx] = row_peak
                        indices[1, idx] = col_peak

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
                        p=0.99,
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
                grp_config.create_dataset("pixel_resolution", data=pixel_resolution)
                grp_config.create_dataset("camera_length", data=clen)


if __name__ == "__main__":
    main()
