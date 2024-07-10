import numpy as np
from dataclasses import dataclass, field
import math
import re
from om.algorithms.crystallography import TypePeakList, Peakfinder8PeakDetection
from om.lib.geometry import (
    TypePixelMaps,
    TypeDetectorLayoutInformation,
    GeometryInformation,
    _read_crystfel_geometry_from_text,
)


@dataclass
class PF8Info:
    max_num_peaks: np.int32 = 200
    adc_threshold: np.int32 = 0
    minimum_snr: np.float32 = 5
    min_pixel_count: np.int16 = 2
    max_pixel_count: np.int16 = 2000
    local_bg_radius: np.int16 = 3
    min_res: np.int16 = 0
    max_res: np.int16 = 1200
    pf8_detector_info: TypeDetectorLayoutInformation = None
    bad_pixel_map_filename: str = None
    bad_pixel_map_hdf5_path: str = None
    pixel_maps: TypePixelMaps = None
    pixel_resolution: float = None
    _shifted_pixel_maps: bool = False
    geometry_txt: list = None

    def update_pixel_maps(self, detector_shift_x: int, detector_shift_y: int):
        if not self._shifted_pixel_maps:
            self._detector_shift_x = detector_shift_x
            self._detector_shift_y = detector_shift_y
            self._shifted_pixel_maps = True
            self.pixel_maps["x"] = (
                self.pixel_maps["x"].flatten() - detector_shift_x
            ).reshape(self._data_shape)
            self.pixel_maps["y"] = (
                self.pixel_maps["y"].flatten() - detector_shift_y
            ).reshape(self._data_shape)
            self.pixel_maps["radius"] = np.sqrt(
                np.square(self.pixel_maps["x"]) + np.square(self.pixel_maps["y"])
            ).reshape(self._data_shape)
            self.pixel_maps["phi"] = np.arctan2(
                self.pixel_maps["y"], self.pixel_maps["x"]
            )
        else:
            raise ValueError(
                f"Pixel maps have been moved once before, to avoid errors reset the geometry before moving it again."
            )

    def set_geometry_from_file(self, geometry_filename: str = None):
        if geometry_filename:
            self.geometry_txt = open(geometry_filename, "r").readlines()
        else:
            if not self.geometry_txt:
                raise ValueError(
                    "Please, specify the detector geometry in CrystFEL format."
                )

        # Passing bad pixel maps to PF8.
        # Warning! It will look for campus in the geom file either 'mask0_file' or 'mask_file'.
        # It doesn't look for multiple masks.
        # It assumes bad pixels as zeros and good pixels as ones.
        try:
            self.bad_pixel_map_filename = [
                x.split(" = ")[-1][:-1]
                for x in self.geometry_txt
                if x.split(" = ")[0] == "mask0_file"
            ][0]
        except IndexError:
            self.bad_pixel_map_filename = [
                x.split(" = ")[-1][:-1]
                for x in self.geometry_txt
                if x.split(" = ")[0] == "mask_file"
            ][0]

        try:
            self.bad_pixel_map_hdf5_path = [
                x.split(" = ")[-1][:-1]
                for x in self.geometry_txt
                if x.split(" = ")[0] == "mask0_data"
            ][0]
        except IndexError:
            self.bad_pixel_map_hdf5_path = [
                x.split(" = ")[-1][:-1]
                for x in self.geometry_txt
                if x.split(" = ")[0] == "mask"
            ][0]

        geom = GeometryInformation(
            geometry_description=self.geometry_txt, geometry_format="crystfel"
        )
        self.pixel_resolution = 1 / geom.get_pixel_size()
        self.pixel_maps = geom.get_pixel_maps()
        self._data_shape = self.pixel_maps["x"].shape
        self._flattened_data_shape = self.pixel_maps["x"].flatten().shape[0]
        self.pf8_detector_info = geom.get_layout_info()
        self._shifted_pixel_maps = False
        self.detector_center_from_geom = self.get_detector_center()

        if (
            self.pf8_detector_info["nasics_x"] * self.pf8_detector_info["nasics_y"]
        ) == 1:
            ## Get single panel transformation matrix from the geometry file
            ### Warning! Check carefully if the visualized data after reorientation of the panel makes sense, e.g. if it is equal to the real experimental data geometry.
            detector, _, _ = _read_crystfel_geometry_from_text(
                text_lines=self.geometry_txt
            )
            detector_panels = dict(detector["panels"])
            panel_name = list(detector_panels.keys())[0]
            frame_dim_structure = [
                x
                for x in detector_panels[panel_name]["dim_structure"]
                if x == "ss" or x == "fs"
            ]
            if frame_dim_structure[0] == "ss":
                self.ss_in_rows = True
            else:
                self.ss_in_rows = False

            fs_string = [
                x.split(" = ")[-1][:-1]
                for x in self.geometry_txt
                if (x.split(" = ")[0]).split("/")[-1] == "fs"
            ][0]

            ss_string = [
                x.split(" = ")[-1][:-1]
                for x in self.geometry_txt
                if (x.split(" = ")[0]).split("/")[-1] == "ss"
            ][0]
            pattern = r"([-+]?\d*\.?\d+)(?=[xyz])"

            try:
                fsx, fsy, fsz = re.findall(pattern, fs_string)
            except ValueError:
                fsx, fsy = re.findall(pattern, fs_string)

            try:
                ssx, ssy, ssz = re.findall(pattern, ss_string)
            except ValueError:
                ssx, ssy = re.findall(pattern, ss_string)

            ## The transformation matrix here is only for visualization purposes. Small stretching factors won't have an impact on the visualization of the images (slabby data).
            self.transformation_matrix = [
                [
                    np.round(float(fsx)),
                    np.round(float(fsy)),
                ],
                [
                    np.round(float(ssx)),
                    np.round(float(ssy)),
                ],
            ]

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

    def get_detector_center(self) -> list:
        if not self._shifted_pixel_maps:

            if (
                self.pf8_detector_info["nasics_x"] * self.pf8_detector_info["nasics_y"]
                > 1
            ):
                # Multiple panels
                # Get minimum array shape
                y_minimum = (
                    2
                    * int(
                        max(
                            abs(self.pixel_maps["y"].max()),
                            abs(self.pixel_maps["y"].min()),
                        )
                    )
                    + 2
                )
                x_minimum = (
                    2
                    * int(
                        max(
                            abs(self.pixel_maps["x"].max()),
                            abs(self.pixel_maps["x"].min()),
                        )
                    )
                    + 2
                )
                visual_img_shape = (y_minimum, x_minimum)
                # Detector center in the middle of the minimum array
                _img_center_x = int(visual_img_shape[1] / 2)
                _img_center_y = int(visual_img_shape[0] / 2)
            else:
                # Single panel
                _img_center_x = int(abs(np.min(self.pixel_maps["x"])))
                _img_center_y = int(abs(np.min(self.pixel_maps["y"])))
        else:
            print(
                "Warning! The detector center was moved by a previous operation, the detector center is not the same as in the geometry file."
            )
            _img_center_x = self.detector_center_from_geom[0] + self._detector_shift_x
            _img_center_y = self.detector_center_from_geom[1] + self._detector_shift_y
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
