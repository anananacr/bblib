from bblib.methods import FriedelPairs
from bblib.models import PF8Info
import h5py
import numpy as np

config = {
    "plots_flag": False,
    "search_radius": 10,
    "polarization": {
        "apply_polarization_correction": False,
        "axis": "x",
        "value": 0.99,
    },
}

plots_info = {
    "filename": "test",
    "folder_name": "test_friedel_pairs",
    "root_path": "/Users/minoru/Documents/ana/scripts/bblib/tests",
    "value_auto": False,
    "value_max": 100,
    "value_min": 1,
    "axis_lim_auto": True,
    "xlim_min": 0,
    "xlim_max": 0,
    "ylim_min": 0,
    "ylim_max": 0,
    "color_map": "viridis",
    "marker_size": 100,
}

PF8Config = PF8Info()


def test_type_output_from_friedel_pairs_without_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    friedel_pairs_method = FriedelPairs(
        config=config, PF8Config=PF8Config, plots_info=plots_info
    )
    with h5py.File("example/thin_glowing_ring_bragg_center_600_200.h5", "r") as f:
        data = np.array(f["image"][0])
    center_from_friedel_pairs = friedel_pairs_method(
        data=data, initial_guess=[604, 202]
    )
    assert isinstance(center_from_friedel_pairs, list)


def test_type_output_from_friedel_pairs_with_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    config["plots_flag"] = True
    friedel_pairs_method = FriedelPairs(
        config=config, PF8Config=PF8Config, plots_info=plots_info
    )
    with h5py.File("example/thin_glowing_ring_bragg_center_600_200.h5", "r") as f:
        data = np.array(f["image"][0])
    center_from_friedel_pairs = friedel_pairs_method(
        data=data, initial_guess=[604, 202]
    )
    assert isinstance(center_from_friedel_pairs, list)


def test_output_from_friedel_pairs_without_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    friedel_pairs_method = FriedelPairs(
        config=config, PF8Config=PF8Config, plots_info=plots_info
    )
    with h5py.File("example/thin_glowing_ring_bragg_center_600_200.h5", "r") as f:
        data = np.array(f["image"][0])
    center_from_friedel_pairs = friedel_pairs_method(
        data=data, initial_guess=[604, 202]
    )
    assert center_from_friedel_pairs == [600, 200]


def test_output_from_friedel_pairs_with_plot():
    PF8Config.set_geometry_from_file("example/simple.geom")
    config["plots_flag"] = True
    friedel_pairs_method = FriedelPairs(
        config=config, PF8Config=PF8Config, plots_info=plots_info
    )
    with h5py.File("example/thin_glowing_ring_bragg_center_600_200.h5", "r") as f:
        data = np.array(f["image"][0])
    center_from_friedel_pairs = friedel_pairs_method(
        data=data, initial_guess=[604, 202]
    )
    assert center_from_friedel_pairs == [600, 200]
