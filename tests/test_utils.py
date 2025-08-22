from bblib.utils import get_fwhm_map_min_from_projection
import numpy as np
import os
from csv import DictReader
from bblib.utils import get_fwhm_map_min_from_projection

def test_fwhm_map_min_from_projection():
    with open("example/grid_search.csv", "r") as csv_file:
        lines = list(DictReader(csv_file))

    center = get_fwhm_map_min_from_projection(lines = lines, label = "test", output_folder = ".", pixel_step = 1, plots_flag = True)
    assert  isinstance(center, list)
