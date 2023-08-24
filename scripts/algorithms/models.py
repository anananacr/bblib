from typing import List, Optional, Callable, Tuple, Any, Dict
import numpy as np
from dataclasses import dataclass, field
import math
from om.algorithms.crystallography import TypePeakList, Peakfinder8PeakDetection

## Some abstractions to call peakfinder8 in Python


def build_pixel_map(row: int, col: int, x0: int, y0: int) -> Dict[str, int]:
    """
    Calculate radius pixels map for a given center x0,y0.

    Parameters
    ----------
    row: int
        Number of rows of data.
    col: int
        Number of columns of data.
    x0:
        Radial center position in x axis (column).
    y0:
        Radial center position in y axis (row).

    Returns
    ----------
    radius_pixel_map: Dict
        "radius": radius pixel map in realtion to the give center. It has same size of data given by row and col.
    """

    [X, Y] = np.meshgrid(np.arange(col) - x0, np.arange(row) - y0)
    R = np.sqrt(np.square(X) + np.square(Y))
    Rint = np.rint(R).astype(int)
    return dict(radius=Rint)


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
    _bad_pixel_map: np.array = None
    _pixelmaps: np.array = field(init=False)

    def modify_radius(self, center_x, center_y):
        self._pixelmaps = build_pixel_map(
            (self._bad_pixel_map).shape[0],
            (self._bad_pixel_map.shape[1]),
            center_x,
            center_y,
        )


class PF8:
    def __init__(self, info):
        assert isinstance(
            info, PF8Info
        ), f"Info object expected type PF8Info, found {type(info)}."
        self.pf8_param = info

    def get_peaks_pf8(self, data):
        detector_layout = self.pf8_param.pf8_detector_info

        peak_detection = Peakfinder8PeakDetection(
            self.pf8_param.max_num_peaks,
            self.pf8_param.pf8_detector_info["asic_nx"],
            self.pf8_param.pf8_detector_info["asic_ny"],
            self.pf8_param.pf8_detector_info["nasics_x"],
            self.pf8_param.pf8_detector_info["nasics_y"],
            self.pf8_param.adc_threshold,
            self.pf8_param.minimum_snr,
            self.pf8_param.min_pixel_count,
            self.pf8_param.max_pixel_count,
            self.pf8_param.local_bg_radius,
            self.pf8_param.min_res,
            self.pf8_param.max_res,
            self.pf8_param._bad_pixel_map.astype(np.float32),
            (self.pf8_param._pixelmaps["radius"]).astype(np.float32),
        )
        peaks_list = peak_detection.find_peaks(data)
        return peaks_list
