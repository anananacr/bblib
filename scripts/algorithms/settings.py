import yaml
from pathlib import Path

def read(path:str):
    path=Path(path)
    config=path.read_text()
    config=yaml.safe_load(config)
    return config

def parse(config:dict):

    return {
    "pf8_max_num_peaks": config["pf8"]["max_num_peaks"],
    "pf8_adc_threshold": config["pf8"]["adc_threshold"],
    "pf8_minimum_snr": config["pf8"]["minimum_snr"],
    "pf8_min_pixel_count": config["pf8"]["min_pixel_count"],
    "pf8_max_pixel_count": config["pf8"]["max_pixel_count"],
    "pf8_local_bg_radius": config["pf8"]["local_bg_radius"],
    "pf8_min_res": config["pf8"]["min_res"],
    "pf8_max_res": config["pf8"]["max_res"],
    "min_peak_region": config["peak_region"]["min"],
    "max_peak_region": config["peak_region"]["max"],
    "canny_sigma": config["canny"]["sigma"],
    "canny_low_thr": config["canny"]["low_threshold"],
    "canny_high_thr": config["canny"]["high_threshold"],
    "outlier_distance": config["outlier_distance"],
    "search_radius": config["search_radius"],
    "method": config["method"],
    "bragg_peaks_positions_for_center_of_mass_calculation": config["bragg_pos_center_of_mass"],
    "pixels_for_mask_of_bragg_peaks": config["pixels_per_peak"],
    "skipped_methods": config["skip_method"],
    "skipped_polarization": config["skip_pol"],
    "offset_x": config["offset"]["x"],
    "offset_y": config["offset"]["y"],
    "force_center_mode": config["force_center"]["mode"],
    "force_center": [config["force_center"]["x"],config["force_center"]["y"]]
    }
