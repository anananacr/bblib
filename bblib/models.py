from typing import List, Optional, Callable, Tuple, Any, Dict
import numpy as np
from dataclasses import dataclass, field
import math
from om.algorithms.crystallography import TypePeakList, Peakfinder8PeakDetection


@dataclass
class PF8Info:
    max_num_peaks: np.int32
    adc_threshold: np.int32
    minimum_snr: np.float32
    min_pixel_count: np.int16
    max_pixel_count: np.int16
    local_bg_radius: np.int16
    min_res: np.int16
    max_res: np.int16
    pf8_detector_info: dict = None
    bad_pixel_map_filename: str = None
    bad_pixel_map_hdf5_path: str = None
    pixel_maps: dict = None
    _shifted_pixel_maps: bool = False

    def modify_radius(self, detector_shift_x: int, detector_shift_y: int):
        if not self._shifted_pixel_maps:
            self._detector_center_from_geom =self.get_detector_center()
            self._detector_shift_x = detector_shift_x
            self._detector_shift_y = detector_shift_y
            self._shifted_pixel_maps = True
            self._data_shape = self.pixel_maps["x"].shape
            self._flattened_data_shape = self.pixel_maps["x"].flatten().shape[0]
            self.pixel_maps["x"] = (
                self.pixel_maps["x"].flatten() - detector_shift_x
            ).reshape(self._data_shape)
            self.pixel_maps["y"] = (
                self.pixel_maps["y"].flatten() - detector_shift_y
            ).reshape(self._data_shape)
            self.pixel_maps["radius"] = np.sqrt(np.square(self.pixel_maps["x"]) + np.square(self.pixel_maps["y"])).reshape(self._data_shape)
        else: 
            print("Pixel maps have been moved once before, to avoid errors reset the geometry before moving it again.")

    def    get(self, parameter: str):
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

    def get_detector_center(self) ->list:
        if not self._shifted_pixel_maps:
            if self.pf8_detector_info["nasics_x"] * self.pf8_detector_info["nasics_y"] > 1:
                # Multiple panels
                # Get minimum array shape
                y_minimum = (
                    2 * int(max(abs(self.pixel_maps["y"].max()), abs(self.pixel_maps["y"].min()))) + 2
                )
                x_minimum = (
                    2 * int(max(abs(self.pixel_maps["x"].max()), abs(self.pixel_maps["x"].min()))) + 2
                )
                visual_img_shape = (y_minimum, x_minimum)
                # Detector center in the middle of the minimum array
                _img_center_x = int(visual_img_shape[1] / 2)
                _img_center_y = int(visual_img_shape[0] / 2)
            else:
                # Single panel
                _img_center_x = int(abs(self.pixel_maps["x"][0, 0]))
                _img_center_y = int(abs(self.pixel_maps["y"][0, 0]))
        else:
            _img_center_x = self._detector_center_from_geom[0] + self._detector_shift_x
            _img_center_y = self._detector_center_from_geom[1] + self._detector_shift_y
        return [_img_center_x, _img_center_y]
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
