from abc import ABC, abstractmethod
import om.lib.geometry as geometry
from utils import (
    mask_peaks,
    center_of_mass,
    azimuthal_average,
    gaussian_lin,
    open_fwhm_map_global_min,
)
from models import PF8Info, PF8
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import multiprocessing


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


class CenterOfMass(CenteringMethod):
    def __init__(self, config: dict, PF8Config: PF8Info):
        self.config = config
        self.PF8Config = PF8Config

    def _prep_for_centering(self, data: np.ndarray, initial_center: tuple) -> None:
        self.initial_center = initial_center
        pf8 = PF8(self.PF8Config)
        peak_list = pf8.get_peaks_pf8(data=data)
        peak_list_x_in_frame, peak_list_y_in_frame = pf8.peak_list_in_slab(peak_list)
        indices = np.ndarray((2, peak_list["num_peaks"]), dtype=int)

        for idx, k in enumerate(peak_list_y_in_frame):
            row_peak = int(k + initial_center[1])
            col_peak = int(peak_list_x_in_frame[idx] + initial_center[0])
            indices[0, idx] = row_peak
            indices[1, idx] = col_peak

        # Assemble data and mask
        data_visualize = geometry.DataVisualizer(pixel_maps=self.PF8Config.pixel_maps)

        with h5py.File(f"{self.PF8Config.bad_pixel_map_filename}", "r") as f:
            mask = np.array(f[f"{self.PF8Config.bad_pixel_map_hdf5_path}"])

        self.corrected_data = data_visualize.visualize_data(data=data * mask)
        visual_mask = data_visualize.visualize_data(data=mask).astype(int)

        # JF for safety
        visual_mask[np.where(self.corrected_data < 0)] = 0

        # Mask Bragg peaks
        peaks_mask = mask_peaks(
            visual_mask,
            indices,
            bragg=self.config["bragg_pos_center_of_mass"],
            n=self.config["pixels_per_peak"],
        )
        self.mask_for_center_of_mass = peaks_mask * visual_mask

    def _run_centering(self, **kwargs) -> tuple:
        converged, center = center_of_mass(
            self.corrected_data, self.mask_for_center_of_mass
        )
        if converged == 0:
            center = self.initial_center.copy()
        center[0] += self.config["offset"]["x"]
        center[1] += self.config["offset"]["y"]
        return converged, np.round(center, 1)


class CircleDetection(CenteringMethod):
    def __init__(self, config: dict, PF8Config: PF8Info, plots_info: dict):
        self.config = config
        self.PF8Config = PF8Config
        self.plots_info = plots_info

    def _prep_for_centering(self, data: np.ndarray, initial_center: tuple) -> None:
        self.initial_center = initial_center
        ## Find peaks
        pf8 = PF8(self.PF8Config)
        peak_list = pf8.get_peaks_pf8(data=data)
        peak_list_x_in_frame, peak_list_y_in_frame = pf8.peak_list_in_slab(peak_list)
        indices = np.ndarray((2, peak_list["num_peaks"]), dtype=int)

        for idx, k in enumerate(peak_list_y_in_frame):
            row_peak = int(k + initial_center[1])
            col_peak = int(peak_list_x_in_frame[idx] + initial_center[0])
            indices[0, idx] = row_peak
            indices[1, idx] = col_peak

        # Assemble data and mask
        data_visualize = geometry.DataVisualizer(pixel_maps=self.PF8Config.pixel_maps)

        with h5py.File(f"{self.PF8Config.bad_pixel_map_filename}", "r") as f:
            mask = np.array(f[f"{self.PF8Config.bad_pixel_map_hdf5_path}"])

        self.corrected_data = data_visualize.visualize_data(data=data * mask)
        visual_mask = data_visualize.visualize_data(data=mask).astype(int)

        # JF for safety
        visual_mask[np.where(self.corrected_data < 0)] = 0
        only_peaks_mask = mask_peaks(
            visual_mask, indices, bragg=0, n=self.config["pixels_per_peak"]
        )
        self.mask_for_circle_detection = only_peaks_mask * visual_mask

    def _run_centering(self, **kwargs) -> tuple:
        ## Scikit-image circle detection
        edges = canny(
            self.corrected_data,
            mask=self.mask_for_circle_detection,
            sigma=self.config["canny"]["sigma"],
            use_quantiles=True,
            low_threshold=self.config["canny"]["low_threshold"],
            high_threshold=self.config["canny"]["high_threshold"],
        )

        if self.config["plots_flag"]:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
            ax1.imshow(edges)
            plt.savefig(
                f'{self.plots_info["args"].scratch}/center_refinement/plots/{self.plots_info["run_label"]}/edges/{self.plots_info["file_label"]}_{self.plots_info["frame_index"]}.png'
            )
            plt.close()
        # Detect radii
        hough_radii = np.arange(
            self.config["peak_region"]["min"], self.config["peak_region"]["max"], 1
        )
        hough_res = hough_circle(edges, hough_radii)
        # Select the most prominent 1 circle
        accums, cx, cy, radii = hough_circle_peaks(
            hough_res, hough_radii, total_num_peaks=1
        )
        if len(cx) > 0:
            center = [cx[0], cy[0]]
            converged = 1
        else:
            center = self.initial_center.copy()
            converged = 0

        return converged, center


class MinimizePeakFWHM(CenteringMethod):
    def __init__(self, config: dict, PF8Config: PF8Info, plots_info: dict):
        self.config = config
        self.PF8Config = PF8Config
        self.plots_info = plots_info
        self.plot_fwhm_flag = False

    def _calculate_fwhm(self, coordinate: tuple) -> dict:
        center_to_radial_average = coordinate
        try:
            x_all, y_all = azimuthal_average(
                self.corrected_data,
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
        except ZeroDivisionError:
            r_squared = 0
            fwhm = 10000
            popt = []

        ## Display plots
        if self.plot_fwhm_flag and len(popt) > 0:
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
                f'{self.plots_info["args"].scratch}/center_refinement/plots/{self.plots_info["run_label"]}/radial_average/{self.plots_info["file_label"]}_{self.plots_info["frame_index"]}.png'
            )
            plt.close()

        return {
            "xc": center_to_radial_average[0],
            "yc": center_to_radial_average[1],
            "fwhm": fwhm,
            "r_squared": r_squared,
        }

    def _prep_for_centering(self, data: np.ndarray, initial_center: tuple) -> None:

        self.initial_center = initial_center
        ## Find peaks
        pf8 = PF8(self.PF8Config)
        peak_list = pf8.get_peaks_pf8(data=data)
        peak_list_x_in_frame, peak_list_y_in_frame = pf8.peak_list_in_slab(peak_list)
        indices = np.ndarray((2, peak_list["num_peaks"]), dtype=int)

        for idx, k in enumerate(peak_list_y_in_frame):
            row_peak = int(k + initial_center[1])
            col_peak = int(peak_list_x_in_frame[idx] + initial_center[0])
            indices[0, idx] = row_peak
            indices[1, idx] = col_peak

        # Assemble data and mask
        data_visualize = geometry.DataVisualizer(pixel_maps=self.PF8Config.pixel_maps)

        with h5py.File(f"{self.PF8Config.bad_pixel_map_filename}", "r") as f:
            mask = np.array(f[f"{self.PF8Config.bad_pixel_map_hdf5_path}"])

        self.corrected_data = data_visualize.visualize_data(data=data * mask)
        visual_mask = data_visualize.visualize_data(data=mask).astype(int)

        # JF for safety
        visual_mask[np.where(self.corrected_data < 0)] = 0

        only_peaks_mask = mask_peaks(
            visual_mask, indices, bragg=0, n=self.config["pixels_per_peak"]
        )
        self.mask_for_fwhm_min = only_peaks_mask * visual_mask

        self.pixel_step = 1
        xx, yy = np.meshgrid(
            np.arange(
                initial_center[0] - self.config["outlier_distance"],
                initial_center[0] + self.config["outlier_distance"] + 1,
                self.pixel_step,
                dtype=int,
            ),
            np.arange(
                initial_center[1] - self.config["outlier_distance"],
                initial_center[1] + self.config["outlier_distance"] + 1,
                self.pixel_step,
                dtype=int,
            ),
        )
        coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))

        ## TEST avoid effects from secondary peaks
        # ring_mask_array=ring_mask(masked_data,initial_center, config["peak_region"]["min"], config["peak_region"]["max"])
        # masked_data[~ring_mask_array]=0

        pool = multiprocessing.Pool()
        with pool:
            self.fwhm_summary = pool.map(self._calculate_fwhm, coordinates)

    def _run_centering(self, **kwargs) -> tuple:
        converged, center = open_fwhm_map_global_min(
            self.fwhm_summary,
            f'{self.plots_info["args"].scratch}/center_refinement/plots/{self.plots_info["run_label"]}',
            f'{self.plots_info["file_label"]}_{self.plots_info["frame_index"]}',
            self.pixel_step,
            self.config["plots_flag"],
        )

        if converged:
            self.plot_fwhm_flag = True
            self._calculate_fwhm(center)
            self.plot_fwhm_flag = False
        return converged, center


