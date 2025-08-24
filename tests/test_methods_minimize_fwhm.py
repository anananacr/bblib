from bblib.methods import MinimizePeakFWHM
from bblib.models import PF8Info
import h5py
import numpy as np

config = {
    "plots_flag": False,
	"offset": {
		"x": 0,
		"y": 0
		},
	"peak_region":{
		"min": 60,
		"max": 180
		},
	"grid_search_radius": 3,
	"pixels_for_mask_of_bragg_peaks": 2,
	"polarization": {
		"apply_polarization_correction": False,
		"axis": "x",
		"value": 0.99
		}
    }

plots_info = {
    "filename": "test",
	"folder_name": "test_minimize_fwhm",
	"root_path": "/Users/minoru/Documents/ana/scripts/bblib/tests",
	"value_auto": False,
	"value_max": 500,
	"value_min": 1,
	"axis_lim_auto": True,
	"xlim_min": 0,
	"xlim_max": 0,
	"ylim_min": 0,
	"ylim_max": 0,
	"color_map": "viridis",
	"marker_size": 20
}

PF8Config = PF8Info()

def test_type_output_from_minimize_peak_fwhm_without_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    minimize_peak_fwhm_method = MinimizePeakFWHM(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/thick_ring_600_200.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_minimize_peak_fwhm = minimize_peak_fwhm_method(data = data, initial_guess=[602,202])
    assert isinstance(center_from_minimize_peak_fwhm, list)

def test_type_output_from_minimize_peak_fwhm_with_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    config["plots_flag"]=True
    minimize_peak_fwhm_method = MinimizePeakFWHM(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/thick_ring_600_200.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_minimize_peak_fwhm = minimize_peak_fwhm_method(data = data, initial_guess=[602,202])
    assert isinstance(center_from_minimize_peak_fwhm, list)


def test_output_from_minimize_peak_fwhm_without_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    minimize_peak_fwhm_method = MinimizePeakFWHM(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/thick_ring_600_200.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_minimize_peak_fwhm = minimize_peak_fwhm_method(data = data, initial_guess=[602,202])
    assert center_from_minimize_peak_fwhm == [600,200]


def test_output_from_minimize_peak_fwhm_with_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    config["plots_flag"]=True
    minimize_peak_fwhm_method = MinimizePeakFWHM(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/thick_ring_600_200.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_minimize_peak_fwhm = minimize_peak_fwhm_method(data = data, initial_guess=[602,202])
    assert center_from_minimize_peak_fwhm == [600,200]
