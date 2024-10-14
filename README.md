# beambusters library

beambusters library. Python library to refine the detector center of diffraction patterns.

For more information, see: https://github.com/anananacr/beambusters

## Python version

Python 3.10

## Installation
pip install bblib

## Usage

To utilize the methods `CenterOfMass`,  `FriedelPairs`, `MinimizePeakFWHM`  and `CircleDetection` it is required to have two configuration dictionaries, one for PeakFinder8 and another one for this library itself. The following snippet shows the general structure for both (parameters not used in your case can be omitted):

```python
config = {
    "plots_flag": ...,
	"search_radius": ...,
	"pf8": {
		"max_num_peaks": ...,
		"adc_threshold": ...,
		"minimum_snr": ...,
		"min_pixel_count": ...,
		"max_pixel_count": ...,
		"local_bg_radius": ...,
		"min_res": ...,
		"max_res": ...
		},
	"offset": {
		"x": ...,
		"y": ...
		},
	"peak_region":{
		"min": ...,
		"max": ...
		},
	"canny":{
		"sigma": ...,
		"low_threshold": ...,
		"high_threshold": ...
		},
	"hough_rank": ...,
	"bragg_peaks_positions_for_center_of_mass_calculation": ...,
	"pixels_for_mask_of_bragg_peaks": ...,
	"polarization": {
		"apply_polarization_correction": ...,
		"axis": ...,
		"value": ...
		}
}

PF8Info = {
	"max_num_peaks": 
	"adc_threshold": 
	"minimum_snr": ...,
	"min_pixel_count": ...,
	"max_pixel_count": ...,
	"local_bg_radius": ...,
	"min_res": ...,
	"max_res": ...,
	"pf8_detector_info": ...,
	"bad_pixel_map_filename": ...,
	"bad_pixel_map_hdf5_path": ...,
	"pixel_maps": ...,
	"pixel_resolution": ...,
	"_shifted_pixel_maps":...
}
```

The `pf8_detector_info` parameter is a dictionary containing the detector layout information:
```python
pf8_detector_info =  {
	"asic_nx": ...,
	"asic_ny": ...,
	"nasics_x": ...,
	"nasics_y": ...
} 
```

The `pixel_maps` parameter is a dictionary containing the pixel maps numpy array:
```python
pixel_maps =  {
	"x": ...,
	"y": ...,
	"z": ...,
	"radius": ...,
	"phi": ...
} 
```

The methods `FriedelPairs`, `MinimizePeakFWHM` and  `CircleDetection ` need a `plots_info` parameter if you want to save plots:
```python
plots_info =  {
	"file_name": ...,
	"folder_name": ...,
	"root_path": ...,
	"value_auto": ...,
	"value_max": ...,
	"value_min": ...,
	"axis_lim_auto": ...,
	"xlim_min": ...,
	"xlim_max": ...,
	"ylim_min": ...,
	"ylim_max": ...,
	"color_map": ...,
	"marker_size": ...
}
```
To calculate the refined detector center of raw data frame as a numpy array using the following methods: 

```python
from bblib.methods import CenterOfMass
center_of_mass_method = CenterOfMass(config=config, PF8Config=PF8Config, plots_info=plots_info)
center_coordinates_from_center_of_mass = center_of_mass_method(
                        data = ...
                    )
                    
from bblib.methods import CircleDetection
circle_detection_method = CircleDetection(config=config, PF8Config=PF8Config, plots_info=plots_info)
center_coordinates_from_circle_detection = circle_detection_method(
                        data = ...
                    )
``` 

The `FriedelPairs` and `MinimizePeakFWHMmethod` need an initial guess for the refined detector center coordinates ` initial_guess = [x_0, y_0]`

```python          
from bblib.methods import MinimizePeakFWHM
minimize_peak_fwhm_method = MinimizePeakFWHM(
                        config=config, PF8Config=PF8Config, plots_info=plots_info
                    )
center_coordinates_from_minimize_peak_fwhm = minimize_peak_fwhm_method(
                        data = ..., initial_guess = ...
                    )


from bblib.methods import FriedelPairs
friedel_pairs_method = FriedelPairs(
                        config=config, PF8Config=PF8Config, plots_info=plots_info
                    )
center_coordinates_from_friedel_pairs = friedel_pairs_method(
                        data = ..., initial_guess= ...
                    )
```         
## Author:

Ana Carolina Rodrigues (2021 - 2024)

Email: sc.anarodrigues@gmail.com



