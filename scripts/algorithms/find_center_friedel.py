from typing import List, Optional, Callable, Tuple, Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import open_distance_map_fit_min

## Fine search
SearchRadius = 14
OutlierDistance = 7

def remove_repeated_items(pairs_list: list) -> list:
    x_vector = []
    y_vector = []
    unique_pairs = []

    for pair in pairs_list:
        peak_0, peak_1 = pair
        x = peak_0[0] - peak_1[0]
        y = peak_0[1] - peak_1[1]
        if x not in x_vector and y not in y_vector:
            x_vector.append(x)
            y_vector.append(y)
            unique_pairs.append((peak_0, peak_1))
    return unique_pairs


def shift_inverted_peaks_and_calculate_minimum_distance(
    peaks_and_shift: list,
) -> Dict[str, float]:
    peaks_list, inverted_peaks, shift = peaks_and_shift
    shifted_inverted_peaks = [(x + shift[0], y + shift[1]) for x, y in inverted_peaks]
    distance = calculate_pair_distance(peaks_list, shifted_inverted_peaks)

    return {
        "shift_x": shift[0],
        "xc": (shift[0] / 2) + ref_center[0],
        "shift_y": shift[1],
        "yc": (shift[1] / 2) + ref_center[1],
        "d": distance,
    }


def calculate_pair_distance(peaks_list: list, shifted_peaks_list: list) -> float:
    d = [
        math.sqrt((peaks_list[idx][0] - i[0]) ** 2 + (peaks_list[idx][1] - i[1]) ** 2)
        for idx, i in enumerate(shifted_peaks_list)
    ]
    return sum(d)


def select_closest_peaks(peaks_list: list, inverted_peaks: list) -> list:
    peaks = []
    for i in inverted_peaks:
        radius = 1
        found_peak = False
        while not found_peak and radius <= SearchRadius:
            found_peak = find_a_peak_in_the_surrounding(peaks_list, i, radius)
            radius += 1
        if found_peak:
            peaks.append((found_peak, i))
    peaks = remove_repeated_items(peaks)
    return peaks

def find_a_peak_in_the_surrounding(
    peaks_list: list, inverted_peak: list, radius: int
) -> list:
    cut_peaks_list = []
    cut_peaks_list = [
        (
            peak,
            math.sqrt(
                (peak[0] - inverted_peak[0]) ** 2 + (peak[1] - inverted_peak[1]) ** 2
            ),
        )
        for peak in peaks_list
        if math.sqrt(
            (peak[0] - inverted_peak[0]) ** 2 + (peak[1] - inverted_peak[1]) ** 2
        )
        <= radius
    ]
    cut_peaks_list.sort(key=lambda x: x[1])

    if cut_peaks_list == []:
        return False
    else:
        return cut_peaks_list[0][0]

def calculate_center_friedel_pairs(corrected_data: np.ndarray, mask: np.ndarray,peaks_list: list, initial_center: list, plots_flag: bool, output_folder: str, label:str):
    global ref_center
    ref_center=initial_center

    peaks_list_x = [k - ref_center[0] for k in peaks_list["fs"]]
    peaks_list_y = [k - ref_center[1] for k in peaks_list["ss"]]
    peaks = list(zip(peaks_list_x, peaks_list_y))
    inverted_peaks_x = [-1 * k for k in peaks_list_x]
    inverted_peaks_y = [-1 * k for k in peaks_list_y]
    inverted_peaks = list(zip(inverted_peaks_x, inverted_peaks_y))
    pairs_list = select_closest_peaks(peaks, inverted_peaks)
    ## Grid search of shifts around the detector center
    pixel_step = 0.2
    xx, yy = np.meshgrid(
        np.arange(-30, 30.2, pixel_step, dtype=float),
        np.arange(-30, 30.2, pixel_step, dtype=float),
    )
    coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
    peaks_0 = [x for x, y in pairs_list]
    peaks_1 = [y for x, y in pairs_list]
    coordinates_anchor_peaks = [
        [peaks_0, peaks_1, shift] for shift in coordinates
    ]
    distance_summary = []
    for shift in coordinates_anchor_peaks:
        distance_summary.append(
            shift_inverted_peaks_and_calculate_minimum_distance(shift)
        )
    ## Display plots
    ## Minimize distance
    # xc, yc, converged = open_distance_map_global_min(
    #    distance_summary, output_folder, f"{label}", pixel_step
    # )
    ## Fine tune
    xc, yc, converged = open_distance_map_fit_min(
        distance_summary, output_folder, f"{label}", pixel_step
    )
    if math.isnan(xc) or math.isnan(yc):
        converged=0
        h, w = corrected_data.shape
        refined_center = (np.around(w/2, 1), np.around(h/2, 1))
    else:
        converged=1
        refined_center = (np.around(xc, 1), np.around(yc, 1))

    shift_x = 2 * (xc - ref_center[0])
    shift_y = 2 * (yc - ref_center[1])
    if plots_flag:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pos = ax.imshow(corrected_data * mask, vmax=20, cmap="cividis")
        ax.scatter(
            ref_center[0],
            ref_center[1],
            color="lime",
            marker="+",
            s=150,
            label=f"Initial center:({np.round(ref_center[0],1)},{np.round(ref_center[1], 1)})",
        )
        ax.scatter(
            refined_center[0],
            refined_center[1],
            color="r",
            marker="o",
            s=25,
            label=f"Refined center:({refined_center[0]}, {refined_center[1]})",
        )
        ax.set_xlim(0, 1028)
        ax.set_ylim(512, 0)
        plt.title("Center refinement: autocorrelation of Friedel pairs")
        fig.colorbar(pos, shrink=0.6)
        ax.legend()
        plt.savefig(f"{output_folder}/centered_friedel/{label}.png")
        plt.close("all")
    
    original_peaks_x = [
        np.round(k + ref_center[0]) for k in peaks_list_x
    ]
    original_peaks_y = [
        np.round(k + ref_center[1]) for k in peaks_list_y
    ]
    inverted_non_shifted_peaks_x = [
        np.round(k + ref_center[0]) for k in inverted_peaks_x
    ]
    inverted_non_shifted_peaks_y = [
        np.round(k + ref_center[1]) for k in inverted_peaks_y
    ]
    inverted_shifted_peaks_x = [
        np.round(k + ref_center[0] + shift_x) for k in inverted_peaks_x
    ]
    inverted_shifted_peaks_y = [
        np.round(k + ref_center[1] + shift_y) for k in inverted_peaks_y
    ]
    if plots_flag:
        ## Check pairs alignement
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pos = ax.imshow(corrected_data * mask, vmax=20, cmap="cividis")
        ax.scatter(
            original_peaks_x,
            original_peaks_y,
            facecolor="none",
            s=100,
            marker="s",
            edgecolor="red",
            label="original peaks",
        )
        ax.scatter(
            inverted_non_shifted_peaks_x,
            inverted_non_shifted_peaks_y,
            s=100,
            facecolor="none",
            marker="s",
            edgecolor="cyan",
            label="inverted peaks",
            alpha=0.5
        )
        ax.scatter(
            inverted_shifted_peaks_x,
            inverted_shifted_peaks_y,
            facecolor="none",
            s=100,
            marker="D",
            edgecolor="lime",
            label="shift of inverted peaks",
        )
        ax.set_xlim(0, 1028)
        ax.set_ylim(512, 0)
        plt.title("Bragg peaks alignement")
        fig.colorbar(pos, shrink=0.6)
        ax.legend()
        plt.savefig(f"{output_folder}/peaks/{label}.png")
        plt.close()
    original_peaks_x = [k + ref_center[0] for k in peaks_list_x]
    original_peaks_y = [k + ref_center[1] for k in peaks_list_y]
    inverted_non_shifted_peaks_x = [
        k + ref_center[0] for k in inverted_peaks_x
    ]
    inverted_non_shifted_peaks_y = [
        k + ref_center[1] for k in inverted_peaks_y
    ]
    inverted_shifted_peaks_x = [
        k + ref_center[0] + shift_x for k in inverted_peaks_x
    ]
    inverted_shifted_peaks_y = [
        k + ref_center[1] + shift_y for k in inverted_peaks_y
    ]
    if converged==1:
        return [xc, yc]
    else:
        return None