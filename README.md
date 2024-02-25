# beambusters library

Beam swepeing serial crystallography data processing library. Methods implementation for detector center determination based on Friedel pairs inversion symmetry. 

## Python version

Python 3.10.5 (main, Jun 21 2022, 11:18:08) [GCC 4.8.5 20150623 (Red Hat 4.8.5-44)] on linux


## Usage

To utilize the methods `CenterOfMass`,  `FriedelPairs`, `MinimizePeakFWHM`  and `CircleDetection` it is required to have two configuration dictionaries, one for PeakFinder8 and another one for this library itself. The follow snippet shows the expected structure for both:
```python
config = {
	"pf8_max_num_peaks": ...,
	"pf8_adc_threshold": ...,
	"pf8_minimum_snr": ...,
	"pf8_min_pixel_count": ...,
	"pf8_max_pixel_count": ...,
	"pf8_local_bg_radius": ...,
	"pf8_min_res": ...,
	"pf8_max_res": ...,
	"min_peak_region": ...,
	"max_peak_region": ...,
	"canny_sigma": ...,
	"canny_low_thr": ...,
	"canny_high_thr": ...,
	"outlier_distance": ...,
	"search_radius": ...,
	"method": ...,
	"bragg_peaks_positions_for_center_of_mass_calculation": ...,
	"pixels_for_mask_of_bragg_peaks": ...,
	"skipped_methods": ...,
	"skipped_polarization": ...,
	"offset_x": ...,
	"offset_y": ...,
	"force_center_mode": ...,
	"force_center": ...,
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

The methods `FriedelPairs`, `MinimizePeakFWHM` and  `CircleDetection ` need a `plots_info` parameter:
```python
plots_info =  {
	"file_label": ...,
    "run_label": ...,
	"frame_index": ...,
    "args": ...
    }
```
To calculate the refined detector center of a frame in numpy array using the methods: 

```python
from methods import CenterOfMass
center_of_mass_method = CenterOfMass(config=config, PF8Config=PF8Config)
center_coordinates_from_center_of_mass = center_of_mass_method(
                        data=frame
                    )
                    
from methods import CircleDetection
circle_detection_method = CircleDetection(config=config, PF8Config=PF8Config, plots_info=plots_info)
center_coordinates_from_circle_detection = circle_detection_method(
                        data=frame
                    )
                    
from methods import MinimizePeakFWHM
minimize_peak_fwhm_method = MinimizePeakFWHM(
                        config=config, PF8Config=PF8Config, plots_info=plots_info
                    )
center_coordinates_from_minimize_peak_fwhm = minimize_peak_fwhm_method(
                        data=frame
                    )
```  

The `FriedelPairs` method need an initial guess for the refined detector center coordinates ` initial_guess = [x_0, y_0]`

```python          
from methods import FriedelPairs
friedel_pairs_method = FriedelPairs(
                        config=config, PF8Config=PF8Config, plots_info=plots_info
                    )
center_coordinates_from_friedel_pairs = friedel_pairs_method(
                        data=frame, initial_guess= ...
                    )
```         
## Author:

Ana Carolina Rodrigues (2021 - )

email: ana.rodrigues@desy.de



