from bblib.methods import CircleDetection
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
		"min": 100,
		"max": 140
		},
	"canny":{
		"sigma": 10,
		"low_threshold": 0.5,
		"high_threshold": 0.99
		},
	"hough_rank": 1,
	"pixels_for_mask_of_bragg_peaks": 2
    }

plots_info = {
    "filename": "test",
	"folder_name": "test_circle",
	"root_path": "/Users/minoru/Documents/ana/scripts/bblib/tests",
	"value_auto": False,
	"value_max": 800,
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

def test_type_output_from_circle_detection_without_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    circle_detection_method = CircleDetection(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/data_random_with_thin_ring.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_circle_detection = circle_detection_method(data = data)
    assert isinstance(center_from_circle_detection, list)

def test_type_output_from_circle_detection_with_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    config["plots_flag"]=True
    circle_detection_method = CircleDetection(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/data_random_with_thin_ring.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_circle_detection = circle_detection_method(data = data)
    assert isinstance(center_from_circle_detection, list)

def test_output_from_circle_detection_without_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    circle_detection_method = CircleDetection(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/data_random_with_thin_ring.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_circle_detection = circle_detection_method(data = data)
    assert center_from_circle_detection == [600,200]


def test_output_from_circle_detection_with_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    config["plots_flag"]=True
    circle_detection_method = CircleDetection(config=config, PF8Config=PF8Config, plots_info=plots_info)
    with h5py.File("example/data_random_with_thin_ring.h5", "r") as f:
        data = np.array(f["data/data"])
    center_from_circle_detection = circle_detection_method(data = data)
    assert center_from_circle_detection == [600,200]
