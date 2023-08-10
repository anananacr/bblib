from typing import List, Optional, Callable, Tuple, Any, Dict
import numpy as np
from dataclasses import dataclass, field
import math
from om.algorithms.crystallography import TypePeakList, Peakfinder8PeakDetection

## Some abstractions to call peakfinder8 in Python


def build_pixel_map(row: int, col: int, y0: int, x0: int) -> Dict[str, int]:
    """
    Calculate radius pixels map for a given center x0,y0.

    Parameters
    ----------
    row: int
        Number of rows of data.
    col: int
        Number of columns of data.
    y0:
        Radial center position in y axis (row).
    x0:
        Radial center position in x axis (column).

    Returns
    ----------
    radius_pixel_map: Dict
        "radius": radius pixel map in realtion to the give center. It has same size of data given by row and col.
    """
    radius_pixel_map = np.ones((row, col)).astype(int)
    for idy, i in enumerate(radius_pixel_map):
        for idx, j in enumerate(i):
            radius_pixel_map[idy, idx] = int(
                math.sqrt((idx - x0) ** 2 + (idy - y0) ** 2)
            )
    return dict(radius=radius_pixel_map)


@dataclass
class PF8Info:
    max_num_peaks: int
    pf8_detector_info: dict
    adc_threshold: float
    minimum_snr: int
    min_pixel_count: int
    max_pixel_count: int
    local_bg_radius: int
    min_res: float
    max_res: float
    _bad_pixel_map: np.array
    _pixelmaps: np.array = field(init=False)

    def __post_init__(self):

        self._pixelmaps = build_pixel_map(
            (self._bad_pixel_map).shape[0],
            (self._bad_pixel_map.shape[1]),
            int((self._bad_pixel_map).shape[0] / 2),
            int((self._bad_pixel_map).shape[1] / 2),
        )

    def modify_mask(self, mask):
        self._bad_pixel_map = mask

    def modify_radius(self, center_x, center_y):
        self._pixelmaps = build_pixel_map(
            (self._bad_pixel_map).shape[0],
            (self._bad_pixel_map.shape[1]),
            center_y,
            center_x,
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
