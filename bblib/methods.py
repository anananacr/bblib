from abc import ABC, abstractmethod
import om.lib.geometry as geometry
from bblib.utils import (
    mask_peaks,
    center_of_mass,
    azimuthal_average,
    gaussian_lin,
    get_fwhm_map_global_min,
    get_distance_map_global_min,
    correct_polarization,
    visualize_single_panel
)
import math
from scipy import ndimage
from bblib.models import PF8Info, PF8
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import multiprocessing
import pathlib
from scipy.optimize import curve_fit

class CenteringMethod(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None: ...

    @abstractmethod
    def _prep_for_centering(self, **kwargs) -> None: ...

    @abstractmethod
    def _run_centering(self) -> tuple: ...

    def __call__(self, **kwargs) -> tuple:
        self._prep_for_centering(**kwargs)
        return self._run_centering()

    def centering_converged(self, center) -> bool:
        if center == [-1, -1]:
            return False
        else:
            return True


class CenterOfMass(CenteringMethod):
    def __init__(self, config: dict, PF8Config: PF8Info,  plots_info: dict = None):
        self.config = config
        self.PF8Config = PF8Config
        self.plots_info = plots_info
        if config["plots_flag"] and not plots_info:
            raise ValueError("From config you want to save plots, please indicate the information to save the plots.")


    def _prep_for_centering(self, data: np.ndarray) -> None:
        self.initial_detector_center = self.PF8Config.get_detector_center()
        pf8 = PF8(self.PF8Config)
        peak_list = pf8.get_peaks_pf8(data=data)
        peak_list_x_in_frame, peak_list_y_in_frame = pf8.peak_list_in_slab(peak_list)
        row_indexes = np.zeros(peak_list["num_peaks"], dtype=int)
        col_indexes = np.zeros(peak_list["num_peaks"], dtype=int)

        for idx, k in enumerate(peak_list_y_in_frame):
            row_peak = int(k + self.initial_detector_center[1])
            col_peak = int(peak_list_x_in_frame[idx] + self.initial_detector_center[0])
            row_indexes[idx] = row_peak
            col_indexes[idx] = col_peak
        peaks_indexes =(row_indexes, col_indexes)
        
        # Assemble data and mask
        data_visualize = geometry.DataVisualizer(pixel_maps=self.PF8Config.pixel_maps)

        with h5py.File(f"{self.PF8Config.bad_pixel_map_filename}", "r") as f:
            mask = np.array(f[f"{self.PF8Config.bad_pixel_map_hdf5_path}"])

        if (self.PF8Config.pf8_detector_info["nasics_x"] * self.PF8Config.pf8_detector_info["nasics_y"]
                > 1
            ):
            self.visual_data = data_visualize.visualize_data(data=data * mask)
            visual_mask = data_visualize.visualize_data(data=mask).astype(int)
        else:
            
            self.visual_data = visualize_single_panel(data, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
            visual_mask = visualize_single_panel(mask, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)

        # JF for safety
        visual_mask[np.where(self.visual_data < 0)] = 0

        # Mask Bragg peaks
        peaks_mask = mask_peaks(
            visual_mask,
            peaks_indexes,
            bragg=self.config["bragg_peaks_positions_for_center_of_mass_calculation"],
            n=self.config["pixels_for_mask_of_bragg_peaks"],
        )
        self.mask_for_center_of_mass = peaks_mask * visual_mask

    def _run_centering(self, **kwargs) -> tuple:
        center = center_of_mass(self.visual_data, self.mask_for_center_of_mass)
        if self.centering_converged(center):
            center[0] += self.config["offset"]["x"]
            center[1] += self.config["offset"]["y"]

        if self.config["plots_flag"]:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
            ax1.imshow(self.visual_data*self.mask_for_center_of_mass, vmax=10, cmap="YlGn", origin="lower")
            ax1.scatter(self.initial_detector_center[0], self.initial_detector_center[1], color='blue', marker='o', label=f"Initial detector center: ({np.round(self.initial_detector_center[0])}, {np.round(self.initial_detector_center[1])})")
            ax1.scatter(center[0], center[1], color='r', marker='o', label=f"Refined detector center: ({center[0]}, {center[1]})")
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/center_of_mass/')
            path.mkdir(parents=True, exist_ok=True)
            ax1.legend()
            plt.savefig(
                f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/center_of_mass/{self.plots_info["file_name"]}.png'
            )
            plt.close()
        return np.round(center, 0)


class CircleDetection(CenteringMethod):
    def __init__(self, config: dict, PF8Config: PF8Info, plots_info: dict =None):
        self.config = config
        self.PF8Config = PF8Config
        self.plots_info = plots_info
        if config["plots_flag"] and not plots_info:
            raise ValueError("From config you want to save plots, please indicate the information to save the plots.")


    def _prep_for_centering(self, data: np.ndarray) -> None:
        self.initial_detector_center = self.PF8Config.get_detector_center()
        ## Find peaks
        pf8 = PF8(self.PF8Config)
        peak_list = pf8.get_peaks_pf8(data=data)
        peak_list_x_in_frame, peak_list_y_in_frame = pf8.peak_list_in_slab(peak_list)
        row_indexes = np.zeros(peak_list["num_peaks"], dtype=int)
        col_indexes = np.zeros(peak_list["num_peaks"], dtype=int)

        for idx, k in enumerate(peak_list_y_in_frame):
            row_peak = int(k + self.initial_detector_center[1])
            col_peak = int(peak_list_x_in_frame[idx] + self.initial_detector_center[0])
            row_indexes[idx] = row_peak
            col_indexes[idx] = col_peak
        peaks_indexes =(row_indexes, col_indexes)
        
        # Assemble data and mask
        data_visualize = geometry.DataVisualizer(pixel_maps=self.PF8Config.pixel_maps)

        with h5py.File(f"{self.PF8Config.bad_pixel_map_filename}", "r") as f:
            mask = np.array(f[f"{self.PF8Config.bad_pixel_map_hdf5_path}"])

        if (self.PF8Config.pf8_detector_info["nasics_x"] * self.PF8Config.pf8_detector_info["nasics_y"]
                > 1
            ):
            self.visual_data = data_visualize.visualize_data(data=data * mask)
            visual_mask = data_visualize.visualize_data(data=mask).astype(int)
        else:
            
            self.visual_data = visualize_single_panel(data, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
            visual_mask = visualize_single_panel(mask, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)

        # JF for safety
        visual_mask[np.where(self.visual_data < 0)] = 0
        only_peaks_mask = mask_peaks(
            visual_mask, peaks_indexes, bragg=0, n=self.config["pixels_for_mask_of_bragg_peaks"]
        )
        self.mask_for_circle_detection = only_peaks_mask * visual_mask

    def _run_centering(self, **kwargs) -> tuple:
        ## Scikit-image circle detection
        edges = canny(
            self.visual_data,
            mask=self.mask_for_circle_detection,
            sigma=self.config["canny"]["sigma"],
            use_quantiles=True,
            low_threshold=self.config["canny"]["low_threshold"],
            high_threshold=self.config["canny"]["high_threshold"],
        )

        if self.config["plots_flag"]:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
            ax1.imshow(edges, origin="lower")
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/edges/')
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/edges/{self.plots_info["file_name"]}.png'
            )
            plt.close()
        # Detect radii
        hough_radii = np.arange(
            self.config["peak_region"]["min"], self.config["peak_region"]["max"], 1
        )
        hough_res = hough_circle(edges, hough_radii)
        # Select the most prominent 1 circle
        accums, xc, yc, radii = hough_circle_peaks(
            hough_res, hough_radii, total_num_peaks=1
        )

        if len(xc) > 0:
            xc = xc[0]
            yc = yc[0]
        else:
            xc = -1
            yc = -1

        center = [xc, yc]
        if self.config["plots_flag"]:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
            ax1.imshow(self.visual_data*self.mask_for_circle_detection, vmax=10, origin="lower", cmap="YlGn")
            ax1.scatter(self.initial_detector_center[0], self.initial_detector_center[1], color='blue', marker='o', label=f"Initial detector center: ({np.round(self.initial_detector_center[0])}, {np.round(self.initial_detector_center[1])})")
            ax1.scatter(center[0], center[1], color='r', marker='o', label=f"Refined detector center: ({center[0]}, {center[1]})")
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/center_circle_detection/')
            path.mkdir(parents=True, exist_ok=True)
            ax1.legend()
            plt.savefig(
                f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/center_circle_detection/{self.plots_info["file_name"]}.png'
            )
            plt.close()
        return center


class MinimizePeakFWHM(CenteringMethod):
    def __init__(self, config: dict, PF8Config: PF8Info, plots_info: dict = None):
        self.config = config
        self.PF8Config = PF8Config
        self.plots_info = plots_info
        self.plot_fwhm_flag = False
        if config["plots_flag"] and not plots_info:
            raise ValueError("From config you want to save plots, please indicate the information to save the plots.")


    def _calculate_fwhm(self, coordinate: tuple) -> dict:
        center_to_radial_average = coordinate
        try:
            x_all, y_all = azimuthal_average(
                self.visual_data,
                center=center_to_radial_average,
                mask=self.mask_for_fwhm_min,
            )
        except IndexError:
            return {
                "xc": center_to_radial_average[0],
                "yc": center_to_radial_average[1],
                "fwhm": 10000,
                "r_squared": 0,
            }

        if self.plot_fwhm_flag:
            fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
            plt.plot(x_all, y_all)

        ## Define background peak region
        x_min = self.config["peak_region"]["min"]
        x_max = self.config["peak_region"]["max"]
        x = x_all[x_min:x_max]
        y = y_all[x_min:x_max]
        ## Estimation of initial parameters

        m0 = 0
        n0 = 2
        y_linear = m0 * x + n0
        y_gaussian = y - y_linear

        try:
            mean = sum(x * y_gaussian) / sum(y_gaussian)
            sigma = np.sqrt(sum(y_gaussian * (x - mean) ** 2) / sum(y_gaussian))
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
        except (ZeroDivisionError, RuntimeError):
            r_squared = 0
            fwhm = 10000
            popt = []

        ## Display plots
        if self.plot_fwhm_flag and len(popt) > 0:
            x_fit = x.copy()
            y_fit = gaussian_lin(x_fit, *popt)

            plt.vlines([x[0], x[-1]], 0, round(popt[0])*10, "r")

            plt.plot(
                x_fit,
                y_fit,
                "r--",
                label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
            )

            plt.title("Azimuthal integration")
            plt.xlim(0, 500)
            plt.ylim(0, 5)
            plt.legend()
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/radial_average/')
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/radial_average/{self.plots_info["file_name"]}.png'
            )
            plt.close()

        return {
            "xc": center_to_radial_average[0],
            "yc": center_to_radial_average[1],
            "fwhm": fwhm,
            "r_squared": r_squared,
        }

    def _prep_for_centering(self, data: np.ndarray, initial_guess: tuple) -> None:
        self.initial_guess = initial_guess
        self.initial_detector_center = self.PF8Config.get_detector_center()
        ## Find peaks
        self.PF8Config.update_pixel_maps(
            initial_guess[0] - self.initial_detector_center[0],
            initial_guess[1] - self.initial_detector_center[1],
        )
        pf8 = PF8(self.PF8Config)

        peak_list = pf8.get_peaks_pf8(data=data)
        peak_list_x_in_frame, peak_list_y_in_frame = pf8.peak_list_in_slab(peak_list)
        row_indexes = np.zeros(peak_list["num_peaks"], dtype=int)
        col_indexes = np.zeros(peak_list["num_peaks"], dtype=int)

        for idx, k in enumerate(peak_list_y_in_frame):
            row_peak = int(k + self.initial_detector_center[1])
            col_peak = int(peak_list_x_in_frame[idx] + self.initial_detector_center[0])
            row_indexes[idx] = row_peak
            col_indexes[idx] = col_peak
        peaks_indexes =(row_indexes, col_indexes)
        
        # Assemble data and mask
        data_visualize = geometry.DataVisualizer(pixel_maps=self.PF8Config.pixel_maps)

        with h5py.File(f"{self.PF8Config.bad_pixel_map_filename}", "r") as f:
            mask = np.array(f[f"{self.PF8Config.bad_pixel_map_hdf5_path}"])

        if self.config["polarization"]["skip"]:
            if (self.PF8Config.pf8_detector_info["nasics_x"] * self.PF8Config.pf8_detector_info["nasics_y"]
                > 1
            ):
                self.visual_data = data_visualize.visualize_data(data=data * mask)
                visual_mask = data_visualize.visualize_data(data=mask).astype(int)
            else:
                self.visual_data = visualize_single_panel(data, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
                visual_mask = visualize_single_panel(mask, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
        else:
            pol_corrected_data, pol_array_map = correct_polarization(
                self.PF8Config.pixel_maps["x"],
                self.PF8Config.pixel_maps["y"],
                float(np.mean(self.PF8Config.pixel_maps["z"]) * self.PF8Config.pixel_resolution),
                data,
                mask=mask,
                polarization_axis=self.config["polarization"]["axis"],
                p=self.config["polarization"]["value"],
            )
            if (self.PF8Config.pf8_detector_info["nasics_x"] * self.PF8Config.pf8_detector_info["nasics_y"]
                > 1
            ):
                self.visual_data = data_visualize.visualize_data(data=pol_corrected_data * mask)
                visual_mask = data_visualize.visualize_data(data=mask).astype(int)
            else:
                self.visual_data = visualize_single_panel(pol_corrected_data, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
                visual_mask = visualize_single_panel(mask, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)


        # JF for safety
        visual_mask[np.where(self.visual_data < 0)] = 0

        only_peaks_mask = mask_peaks(
            visual_mask, peaks_indexes, bragg=0, n=self.config["pixels_for_mask_of_bragg_peaks"]
        )
        self.mask_for_fwhm_min = only_peaks_mask * visual_mask

        self.pixel_step = 1
        xx, yy = np.meshgrid(
            np.arange(
                self.initial_guess[0] - self.config["outlier_distance"],
                self.initial_guess[0] + self.config["outlier_distance"] + 1,
                self.pixel_step,
                dtype=int,
            ),
            np.arange(
                self.initial_guess[1] - self.config["outlier_distance"],
                self.initial_guess[1] + self.config["outlier_distance"] + 1,
                self.pixel_step,
                dtype=int,
            ),
        )
        coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))

        ## TEST avoid effects from secondary peaks
        # ring_mask_array=ring_mask(masked_data,initial_guess config["peak_region"]["min"], config["peak_region"]["max"])
        # masked_data[~ring_mask_array]=0

        pool = multiprocessing.Pool()
        with pool:
            self.fwhm_summary = pool.map(self._calculate_fwhm, coordinates)

    def _run_centering(self, **kwargs) -> tuple:
        if self.config["plots_flag"]:
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/fwhm_map/')
            path.mkdir(parents=True, exist_ok=True)

        center = get_fwhm_map_global_min(
            self.fwhm_summary,
            f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}',
            f'{self.plots_info["file_name"]}',
            self.pixel_step,
            self.config["plots_flag"]
        )

        if self.centering_converged(center):
            self.plot_fwhm_flag = True
            self._calculate_fwhm(center)
            self.plot_fwhm_flag = False

        if self.config["plots_flag"]:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
            ax1.imshow(self.visual_data * self.mask_for_fwhm_min, vmax=10, origin="lower", cmap="YlGn")
            ax1.scatter(self.initial_detector_center[0], self.initial_detector_center[1], color='blue', marker='o', label=f"Initial detector center: ({np.round(self.initial_detector_center[0])}, {np.round(self.initial_detector_center[1])})")
            ax1.scatter(self.initial_detector_center[0], self.initial_detector_center[1], color='blue', marker='o', label=f"Initial guess: ({np.round(self.initial_guess[0])}, {np.round(self.initial_guess[1])})")
            ax1.scatter(center[0], center[1], color='r', marker='o', label=f"Refined detector center: ({center[0]}, {center[1]})")
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/center_fwhm_minimization/')
            path.mkdir(parents=True, exist_ok=True)
            ax1.legend()
            plt.savefig(
                f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/center_fwhm_minimization/{self.plots_info["file_name"]}.png'
            )
            plt.close()

        return center


class FriedelPairs(CenteringMethod):
    def __init__(self, config: dict, PF8Config: PF8Info, plots_info: dict = None):
        self.config = config
        self.PF8Config = PF8Config
        self.plots_info = plots_info
        if config["plots_flag"] and not plots_info:
            raise ValueError("From config you want to save plots, please indicate the information to save the plots.")


    def _remove_repeated_pairs(self, pairs_list: list) -> list:
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

    def _shift_inverted_peaks_and_calculate_minimum_distance(
        self,
        shift: list,
    ) -> dict:
        peaks_list = self.peaks_list_original.copy()
        inverted_peaks = self.peaks_list_inverted.copy()

        shifted_inverted_peaks = [
            (x + shift[0], y + shift[1]) for x, y in inverted_peaks
        ]
        distance = self.calculate_pair_distance(peaks_list, shifted_inverted_peaks)

        return {
            "shift_x": shift[0],
            "xc": (shift[0] / 2) + self.initial_guess[0] + 0.5,
            "shift_y": shift[1],
            "yc": (shift[1] / 2) + self.initial_guess[1] + 0.5,
            "d": distance,
        }

    def calculate_pair_distance(
        self, peaks_list: list, shifted_peaks_list: list
    ) -> float:
        d = [
            math.sqrt(
                (peaks_list[idx][0] - i[0]) ** 2 + (peaks_list[idx][1] - i[1]) ** 2
            )
            for idx, i in enumerate(shifted_peaks_list)
        ]
        return sum(d)

    def _select_closest_peaks(self, peaks_list: list, inverted_peaks: list) -> list:
        peaks = []
        for i in inverted_peaks:
            radius = 1
            found_peak = False
            while not found_peak and radius <= self.config["search_radius"]:
                found_peak = self._find_a_peak_in_the_surrounding(peaks_list, i, radius)
                radius += 1
            if found_peak:
                peaks.append((found_peak, i))
        # peaks = self._remove_repeated_pairs(peaks)
        return peaks

    def _find_a_peak_in_the_surrounding(
        self, peaks_list: list, inverted_peak: list, radius: int
    ) -> list:
        cut_peaks_list = []
        cut_peaks_list = [
            (
                peak,
                math.sqrt(
                    (peak[0] - inverted_peak[0]) ** 2
                    + (peak[1] - inverted_peak[1]) ** 2
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

    def _prep_for_centering(self, data: np.ndarray, initial_guess: tuple) -> None:

        self.initial_guess = initial_guess
        self.initial_detector_center = self.PF8Config.get_detector_center()
        ## Find peaks
        self.PF8Config.update_pixel_maps(
            initial_guess[0] - self.initial_detector_center[0],
            initial_guess[1] - self.initial_detector_center[1],
        )

        pf8 = PF8(self.PF8Config)
        peak_list = pf8.get_peaks_pf8(data=data)
        peak_list_in_slab = pf8.peak_list_in_slab(peak_list)

        self.peak_list_x_in_frame, self.peak_list_y_in_frame = peak_list_in_slab

        # Assemble data and mask
        data_visualize = geometry.DataVisualizer(pixel_maps=self.PF8Config.pixel_maps)

        with h5py.File(f"{self.PF8Config.bad_pixel_map_filename}", "r") as f:
            mask = np.array(f[f"{self.PF8Config.bad_pixel_map_hdf5_path}"])

        if self.config["polarization"]["skip"]:
            if (self.PF8Config.pf8_detector_info["nasics_x"] * self.PF8Config.pf8_detector_info["nasics_y"]
                > 1
            ):
                self.visual_data = data_visualize.visualize_data(data=data * mask)
                visual_mask = data_visualize.visualize_data(data=mask).astype(int)
            else:
                self.visual_data = visualize_single_panel(data, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
                visual_mask = visualize_single_panel(mask, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
        else:
            pol_corrected_data, pol_array_map = correct_polarization(
                self.PF8Config.pixel_maps["x"],
                self.PF8Config.pixel_maps["y"],
                float(np.mean(self.PF8Config.pixel_maps["z"]) * self.PF8Config.pixel_resolution),
                data,
                mask=mask,
                polarization_axis=self.config["polarization"]["axis"],
                p=self.config["polarization"]["value"],
            )
            if (self.PF8Config.pf8_detector_info["nasics_x"] * self.PF8Config.pf8_detector_info["nasics_y"]
                > 1
            ):
                self.visual_data = data_visualize.visualize_data(data=pol_corrected_data * mask)
                visual_mask = data_visualize.visualize_data(data=mask).astype(int)
            else:
                self.visual_data = visualize_single_panel(pol_corrected_data, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)
                visual_mask = visualize_single_panel(mask, self.PF8Config.transformation_matrix, self.PF8Config.ss_in_rows)


    def _run_centering(self, **kwargs) -> tuple:
        peak_list_x_in_frame = self.peak_list_x_in_frame.copy()
        peak_list_y_in_frame = self.peak_list_y_in_frame.copy()

        peaks = list(zip(peak_list_x_in_frame, peak_list_y_in_frame))
        inverted_peaks_x = [-1 * k for k in peak_list_x_in_frame]
        inverted_peaks_y = [-1 * k for k in peak_list_y_in_frame]
        inverted_peaks = list(zip(inverted_peaks_x, inverted_peaks_y))
        pairs_list = self._select_closest_peaks(peaks, inverted_peaks)

        ## Grid search of shifts around the detector center
        self.pixel_step = 0.2
        xx, yy = np.meshgrid(
            np.arange(
                -self.config["outlier_distance"],
                self.config["outlier_distance"] + 0.2,
                self.pixel_step,
                dtype=float,
            ),
            np.arange(
                -self.config["outlier_distance"],
                self.config["outlier_distance"] + 0.2,
                self.pixel_step,
                dtype=float,
            ),
        )
        coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
        self.peaks_list_original = [x for x, y in pairs_list]
        self.peaks_list_inverted = [y for x, y in pairs_list]

        pool = multiprocessing.Pool()
        with pool:
            distance_summary = pool.map(
                self._shift_inverted_peaks_and_calculate_minimum_distance, coordinates
            )

        ## Minimize distance
        if self.config["plots_flag"]:
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/distance_map/')
            path.mkdir(parents=True, exist_ok=True)

        center = get_distance_map_global_min(
            distance_summary,
            f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}',
            f'{self.plots_info["file_name"]}',
            self.pixel_step,
            self.config["plots_flag"],
        )

        if self.config["plots_flag"] and self.centering_converged(center):
            shift_x = 2 * (center[0] - self.initial_guess[0])
            shift_y = 2 * (center[1] - self.initial_guess[1])

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            pos = ax.imshow(self.visual_data, vmin=0, vmax=100, cmap="YlGn", origin="lower")
            ax.scatter(
                self.initial_detector_center[0],
                self.initial_detector_center[1],
                color="b",
                marker="o",
                s=25,
                label=f"Initial detector center:({np.round(self.initial_detector_center[0],1)},{np.round(self.initial_detector_center[1], 1)})",
            )
            ax.scatter(
                self.initial_guess[0],
                self.initial_guess[1],
                color="lime",
                marker="+",
                s=150,
                label=f"Initial guess:({np.round(self.initial_guess[0],1)},{np.round(self.initial_guess[1], 1)})",
            )
            
            ax.scatter(
                center[0],
                center[1],
                color="r",
                marker="o",
                s=25,
                label=f"Refined detector center:({np.round(center[0],1)}, {np.round(center[1],1)})",
            )
            ax.set_xlim(200, 900)
            ax.set_ylim(900, 200)
            plt.title("Center refinement: autocorrelation of Friedel pairs")
            fig.colorbar(pos, shrink=0.6)
            ax.legend()
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/centered_friedel/')
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/centered_friedel/{self.plots_info["file_name"]}.png'
            )
            plt.close("all")

            original_peaks_x = [
                np.round(k + self.initial_guess[0]) for k in peak_list_x_in_frame
            ]
            original_peaks_y = [
                np.round(k + self.initial_guess[1]) for k in peak_list_y_in_frame
            ]

            inverted_non_shifted_peaks_x = [
                np.round(k + self.initial_guess[0]) for k in inverted_peaks_x
            ]
            inverted_non_shifted_peaks_y = [
                np.round(k + self.initial_guess[1]) for k in inverted_peaks_y
            ]
            inverted_shifted_peaks_x = [
                np.round(k + self.initial_guess[0] + shift_x) for k in inverted_peaks_x
            ]
            inverted_shifted_peaks_y = [
                np.round(k + self.initial_guess[1] + shift_y) for k in inverted_peaks_y
            ]

            ## Check pairs alignement
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            pos = ax.imshow(self.visual_data, vmin=0, vmax=100, cmap="YlGn", origin="lower")
            ax.scatter(
                original_peaks_x,
                original_peaks_y,
                facecolor="none",
                s=80,
                marker="s",
                edgecolor="red",
                linewidth=1.5,
                label="original peaks",
            )
            ax.scatter(
                inverted_non_shifted_peaks_x,
                inverted_non_shifted_peaks_y,
                s=80,
                facecolor="none",
                marker="s",
                edgecolor="tab:orange",
                linewidth=1.5,
                label="inverted peaks",
                alpha=0.8,
            )
            ax.scatter(
                inverted_shifted_peaks_x,
                inverted_shifted_peaks_y,
                facecolor="none",
                s=120,
                marker="D",
                linewidth=1.8,
                alpha=0.8,
                edgecolor="blue",
                label="shift of inverted peaks",
            )
            ax.set_xlim(200, 900)
            ax.set_ylim(900, 200)
            plt.title("Bragg peaks alignement")
            fig.colorbar(pos, shrink=0.6)
            ax.legend()
            path = pathlib.Path(f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/peaks/')
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f'{self.plots_info["root_path"]}/center_refinement/plots/{self.plots_info["folder_name"]}/peaks/{self.plots_info["file_name"]}.png'
            )
            plt.close()
        return center
