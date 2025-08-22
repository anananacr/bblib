from bblib.utils import get_fwhm_map_min_from_projection
from csv import DictReader
from bblib.utils import get_fwhm_map_min_from_projection

def test_type_output_from_fwhm_map_min_from_projection_without_plot():
    with open("example/grid_search.csv", "r") as csv_file:
        lines = list(DictReader(csv_file))

    center = get_fwhm_map_min_from_projection(lines = lines, label = "", output_folder = "", pixel_step = 1, plots_flag = False)
    assert  isinstance(center, list)

def test_type_output_from_fwhm_map_min_from_projection_with_plot():
    with open("example/grid_search.csv", "r") as csv_file:
        lines = list(DictReader(csv_file))

    center = get_fwhm_map_min_from_projection(lines = lines, label = "test", output_folder = ".", pixel_step = 1, plots_flag = True)
    assert  isinstance(center, list)

def test_output_value_from_fwhm_map_min_from_projection_without_plot():
    with open("example/grid_search.csv", "r") as csv_file:
        lines = list(DictReader(csv_file))

    center = get_fwhm_map_min_from_projection(lines = lines, label = "", output_folder = "", pixel_step = 1, plots_flag = False)
    assert  center == [7,13]

def test_output_value_from_fwhm_map_min_from_projection_with_plot():
    with open("example/grid_search.csv", "r") as csv_file:
        lines = list(DictReader(csv_file))

    center = get_fwhm_map_min_from_projection(lines = lines, label = "test", output_folder = ".", pixel_step = 1, plots_flag = True)
    assert  center == [7,13]
