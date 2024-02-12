from typing import List, Optional, Callable, Tuple, Any, Dict
import numpy as np
from dataclasses import dataclass, field
import math
from om.algorithms.crystallography import TypePeakList, Peakfinder8PeakDetection

## Some abstractions to shift detector center for peakfinder8


@dataclass
class PF8Info:
    max_num_peaks: int
    adc_threshold: float
    minimum_snr: int
    min_pixel_count: int
    max_pixel_count: int
    local_bg_radius: int
    min_res: float
    max_res: float
    pf8_detector_info: dict = None
    bad_pixel_map_filename: str = None
    bad_pixel_map_hdf5_path: str = None
    pixel_maps: np.array = None

    def modify_radius(self, detector_shift_x: int, detector_shift_y: int):
        self._data_shape = self.pixel_maps["x"].shape
        self.pixel_maps["x"] = (
            self.pixel_maps["x"].flatten() + detector_shift_x
        ).reshape(self._data_shape)
        self.pixel_maps["y"] = (
            self.pixel_maps["y"].flatten() + detector_shift_y
        ).reshape(self._data_shape)

    def get(self, parameter: str):
        if parameter == "max_num_peaks":
            return self.max_num_peaks
        elif parameter == "adc_threshold":
            return self.adc_threshold
        elif parameter == "minimum_snr":
            return self.minimum_snr
        elif parameter == "min_pixel_count":
            return self.min_pixel_count
        elif parameter == "max_pixel_count":
            return self.max_pixel_count
        elif parameter == "local_bg_radius":
            return self.local_bg_radius
        elif parameter == "min_res":
            return self.min_res
        elif parameter == "max_res":
            return self.max_res
        elif parameter == "bad_pixel_map_filename":
            return self.bad_pixel_map_filename
        elif parameter == "bad_pixel_map_hdf5_path":
            return self.bad_pixel_map_hdf5_path


class PF8:
    def __init__(self, info):
        assert isinstance(
            info, PF8Info
        ), f"Info object expected type PF8Info, found {type(info)}."
        self.pf8_param = info

    def get_peaks_pf8(self, data):
        self._radius_pixel_map = self.pf8_param.pixel_maps["radius"]
        self._data_shape: Tuple[int, ...] = self._radius_pixel_map.shape
        self._flattened_visualization_pixel_map_x = self.pf8_param.pixel_maps[
            "x"
        ].flatten()
        self._flattened_visualization_pixel_map_y = self.pf8_param.pixel_maps[
            "y"
        ].flatten()
        peak_detection = Peakfinder8PeakDetection(
            radius_pixel_map=(self.pf8_param.pixel_maps["radius"]).astype(np.float32),
            layout_info=self.pf8_param.pf8_detector_info,
            crystallography_parameters=self.pf8_param,
        )
        peak_list = peak_detection.find_peaks(data=data)
        return peak_list

    def peak_list_in_slab(self, peak_list):
        ## From OM
        peak_list_x_in_frame: List[float] = []
        peak_list_y_in_frame: List[float] = []
        peak_fs: float
        peak_ss: float
        peak_value: float
        for peak_fs, peak_ss, peak_value, peak_max_pixel_intensity in zip(
            peak_list["fs"],
            peak_list["ss"],
            peak_list["intensity"],
            peak_list["max_pixel_intensity"],
        ):
            peak_index_in_slab: int = int(round(peak_ss)) * self._data_shape[1] + int(
                round(peak_fs)
            )
            y_in_frame: float = self._flattened_visualization_pixel_map_y[
                peak_index_in_slab
            ]
            x_in_frame: float = self._flattened_visualization_pixel_map_x[
                peak_index_in_slab
            ]
            peak_list_x_in_frame.append(x_in_frame)
            peak_list_y_in_frame.append(y_in_frame)

        return peak_list_x_in_frame, peak_list_y_in_frame
