from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import sys
from datetime import datetime
import os
sys.path.append("/home/rodria/software/vdsCsPadMaskMaker/new-versions/")
import geometry_funcs as gf
import argparse
import numpy as np
from utils import (
    get_format,
    mask_peaks,
    gaussian,
    center_of_mass,
    azimuthal_average,
    open_fwhm_map_global_min,
    circle_mask
)
from models import PF8, PF8Info
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
from scipy import signal
import subprocess as sub
import h5py
import hdf5plugin
import multiprocessing
MinPeaks = 10

PF8Config = PF8Info(
    max_num_peaks=10000,
    adc_threshold=0,
    minimum_snr=7,
    min_pixel_count=1,
    max_pixel_count=30,
    local_bg_radius=4,
    min_res=0,
    max_res=1200
)

PixelResolution=1/(75*1e-3)
AutoFlag=False
ForceCenter=[587,400]
OutlierDistance=80
#ForceCenter=[636,400]
#OutlierDistance=10
BadPixelValue=4e9
MinPeakRegion=200
MaxPeakRegion=350

BeamSweepingParam = {
    "pixel_resolution": PixelResolution,
    "min_peaks": MinPeaks,
    "pf8_max_num_peaks": PF8Config.max_num_peaks,
    "pf8_adc_threshold": PF8Config.adc_threshold,
    "pf8_minimum_snr": PF8Config.minimum_snr,
    "pf8_min_pixel_count": PF8Config.min_pixel_count,
    "pf8_max_pixel_count": PF8Config.max_pixel_count,
    "pf8_local_bg_radius": PF8Config.local_bg_radius,
    "pf8_min_res": PF8Config.min_res,
    "pf8_max_res": PF8Config.max_res,
    "min_peak_region": MinPeakRegion,
    "max_peak_region": MaxPeakRegion,
    "bad_pixel_value":BadPixelValue,
    "auto_flag":AutoFlag
}

def calculate_fwhm(data_and_coordinates: tuple) -> Dict[str, int]:
    corrected_data, mask, center_to_radial_average = data_and_coordinates
    x, y = azimuthal_average(
        corrected_data, center=center_to_radial_average, mask=mask
    )
    x_all = x.copy()
    y_all = y.copy()
    
    if plot_flag:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(x, y)
    
    ## Define background peak region
    x_min = MinPeakRegion
    x_max = MaxPeakRegion
    x = x[x_min:x_max]
    y = y[x_min:x_max]
    ## Estimation of initial parameters
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
        residuals = y - gaussian(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ## Calculation of FWHM
        fwhm = popt[2] * math.sqrt(8 * np.log(2))
        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]
    except:
        popt=[max(y), mean, sigma]
        r_squared = 1
        fwhm = 800
        fwhm_over_radius = 800

    ## Display plots
    if plot_flag:
        
        x_fit = x.copy()
        y_fit = gaussian(x_fit, *popt)
        
        plt.vlines([x[0], x[-1]], 0,round(popt[0])+2, "r")

        plt.plot(
            x_fit,
            y_fit,
            "r--",
            label=f"gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM : {round(fwhm,3)}",
        )
        
        plt.title("Azimuthal integration")
        plt.xlim(0, 500)
        #plt.ylim(0, round(popt[0])+2)
        plt.legend()
        plt.savefig(
            f"{args.scratch}/center_refinement/plots/radial_average/{run_label}/{file_label}_{frame_index}.png"
        )
        # plt.show()
        plt.close()

    return {
        "xc": center_to_radial_average[0],
        "yc": center_to_radial_average[1],
        "fwhm": fwhm,
        "fwhm_over_radius": fwhm_over_radius,
        "r_squared": r_squared,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate center of diffraction patterns fro MHz beam sweeping serial crystallography."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )

    parser.add_argument(
        "-m", "--mask", type=str, action="store", help="path to list of mask files .lst"
    )

    parser.add_argument(
        "-g",
        "--geom",
        type=str,
        action="store",
        help="CrystFEL geometry filename",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )
    parser.add_argument(
        "-s", "--scratch", type=str, action="store", help="path to output data files"
    )

    global args
    args = parser.parse_args()
    global plot_flag
    plot_flag = False

    global frame_index

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    
    file_format = get_format(args.input)
    
    ### Extract geometry file
    x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
        args.geom, return_dict=True
    )
    preamb, dim_info = gf.read_geometry_file_preamble(args.geom)
    dist_m = preamb["coffset"]
    res = preamb["res"]
    clen = preamb["clen"]
    dist = 0.0 
    print(det_dict)
    ## Alexandra Tolstikova's code
    _img_center_x=det_dict["panel0"]["corner_y"]
    _img_center_y=-1*det_dict["panel0"]["corner_x"]
    DetectorCenter=[_img_center_x, _img_center_y]

    print(DetectorCenter)
    """"
    if clen is not None:
        if not gf.is_float_try(clen):
            check = H5_name + clen
            myCmd = os.popen("h5ls " + check).read()
            if "NOT" in myCmd:
                # print("Error: no clen from .h5 file")
                clen_v = 0.0
            else:
                f = h5py.File(H5_name, "r")
                clen_v = f[clen][()] * (1e-3)  # f[clen].value * (1e-3)
                f.close()
                pol_bool = True
                # print("Take into account polarisation")
        else:
            clen_v = float(clen)
            pol_bool = True
            # print("Take into account polarisation")
        if dist_m is not None:
            dist_m += clen_v
        else:
            # print("Error: no coffset in geometry file. It is considered as 0.")
            dist_m = 0.0
        # print("CLEN, COFSET", clen, dist_m)
        dist = dist_m * res
    """

    f = h5py.File(f"{args.mask}", "r")
    mask=np.array(f["data/data"])
    f.close()

    global file_label
    global run_label
    if file_format == "lst":
        for i in range(0, len(paths[:])):
            hit_list=[]
            initial_center_list=[]
            center_list=[]
            shift_x_mm_list=[]
            shift_y_mm_list=[]
            
            file_name_str = str(paths[i][:-1])     
            split_path=(os.path.dirname(file_name_str)).split('/')
            #run_label=split_path[-3]+'/'+split_path[-2]+'/'+split_path[-1]
            run_label=split_path[-2]+'/'+split_path[-1]

            # create run folder in processed folder, scratch_cc centered and fwhm_map
            file_label=os.path.basename(file_name_str).split('.')[0]

            if not os.path.exists(f"{args.output}/centered/{run_label}"):
            #if True:

                #cmd=f"mkdir {args.output}/centered/{split_path[-3]}/;mkdir {args.output}/centered/{split_path[-3]}/{split_path[-2]};"
                cmd=f"mkdir {args.output}/centered/{split_path[-2]}; mkdir {args.output}/centered/{run_label}"
                sub.call(cmd, shell=True)
                cmd=f"mkdir {args.scratch}/center_refinement/plots/radial_average/{split_path[-2]}; mkdir {args.scratch}/center_refinement/plots/radial_average/{run_label}"
                sub.call(cmd, shell=True)
                cmd=f"mkdir {args.scratch}/center_refinement/plots/centered/{split_path[-2]}/; mkdir {args.scratch}/center_refinement/plots/centered/{run_label};"
                sub.call(cmd, shell=True)
                cmd=f"mkdir {args.scratch}/center_refinement/plots/fwhm_map/{split_path[-2]}/; mkdir {args.scratch}/center_refinement/plots/fwhm_map/{run_label};"
                sub.call(cmd, shell=True)
                cmd=f"  "
                sub.call(cmd, shell=True)

            file_name = paths[i][:-1]     

            if get_format(file_name) == "h":
                f = h5py.File(f"{file_name}", "r")     
                data = f["entry/data"]["data"] 
            ## Polarization correction factor
            
            for frame_index in range(data.shape[0]):
            #for frame_index in range(4):
                
                frame=np.array(data[frame_index])

                #create_pixel_maps(frame.shape, first_center)
                #corrected_data, pol_array_first = correct_polarization(
                #    x_map, y_map, clen_v, frame, mask=mask
                #)
                corrected_data=frame
                #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                #pos1 = ax1.imshow(frame * mask, vmax=50, cmap="cividis")
                #pos2 = ax2.imshow(corrected_data * mask, vmax=50, cmap="cividis")
                #pos3 = ax3.imshow(pol_array_first, vmin=0.7, vmax=1, cmap="cividis")
                #ax1.set_title("Original data")
                #ax2.set_title("Polarization corrected data")
                #ax3.set_title("Polarization array")
                #fig.colorbar(pos1, shrink=0.6, ax=ax1)
                #fig.colorbar(pos2, shrink=0.6, ax=ax2)
                #fig.colorbar(pos3, shrink=0.6, ax=ax3)
                # plt.show()
                #plt.savefig(f"{args.output}/plots/pol/{label}_{i}.png")
                #plt.close()

                ## Peakfinder8 detector information and bad_pixel_map

                ## Peak search 
                PF8Config.pf8_detector_info = dict(
                    asic_nx=mask.shape[1],
                    asic_ny=mask.shape[0],
                    nasics_x=1,
                    nasics_y=1,
                )
                PF8Config._bad_pixel_map = mask
                PF8Config.modify_radius(
                    DetectorCenter[0], DetectorCenter[1]
                )
                pf8 = PF8(PF8Config)
                peaks_list = pf8.get_peaks_pf8(data=corrected_data)
                indices = (
                    np.array(peaks_list["ss"], dtype=int),
                    np.array(peaks_list["fs"], dtype=int),
                )
                # Mask Bragg  peaks
                only_peaks_mask = mask_peaks(mask, indices, bragg=0)
                first_mask = only_peaks_mask * mask
                
                if AutoFlag:
                    first_center=center_of_mass(corrected_data, first_mask)
                else:
                    first_center=ForceCenter.copy()

                #plt.imshow(corrected_data*mask)
                #plt.scatter(first_center[0], first_center[1],c='r')
                #plt.show()
                initial_center_list.append(first_center)
                
                PF8Config.modify_radius(first_center[0], first_center[1])
                pf8 = PF8(PF8Config)
                peaks_list = pf8.get_peaks_pf8(data=corrected_data)
                print(first_center)
                if peaks_list["num_peaks"] >= MinPeaks:
                    
                    hit_list.append(frame)
                    ## Peak search 
                    PF8Config.pf8_detector_info = dict(
                        asic_nx=mask.shape[1],
                        asic_ny=mask.shape[0],
                        nasics_x=1,
                        nasics_y=1,
                    )
                    PF8Config._bad_pixel_map = mask
                    PF8Config.modify_radius(
                        int(first_center[0]), int(first_center[1])
                    )
                    pf8 = PF8(PF8Config)

                    peaks_list = pf8.get_peaks_pf8(data=corrected_data)
                    indices = (
                        np.array(peaks_list["ss"], dtype=int),
                        np.array(peaks_list["fs"], dtype=int),
                    )

                    # Mask Bragg  peaks
                    only_peaks_mask = mask_peaks(mask, indices, bragg=0)
                    pf8_mask = only_peaks_mask * mask

                    # Brute force manner
                    ## Grid search of sharpness of the azimutal average

                    pixel_step = 1
                    xx, yy = np.meshgrid(
                        np.arange(first_center[0] - OutlierDistance, first_center[0] + OutlierDistance + 1, pixel_step, dtype=int),
                        np.arange(first_center[1] - OutlierDistance, first_center[1] + OutlierDistance + 1, pixel_step, dtype=int)
                    )

                    coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                    coordinates_anchor_data=[(corrected_data, pf8_mask, shift) for shift in coordinates]
                    fwhm_summary=[]
                    #for k in coordinates_anchor_data:
                    #    fwhm_summary.append(calculate_fwhm(k))
                    pool = multiprocessing.Pool()
                    with pool:
                        fwhm_summary = pool.map(calculate_fwhm, coordinates_anchor_data)

                    ## Display plots
                    xc, yc = open_fwhm_map_global_min(
                        fwhm_summary, f"{args.scratch}/center_refinement/plots/fwhm_map/{run_label}/{file_label}_{frame_index}", pixel_step
                    )

                    refined_center = (int(np.round(xc)), int(np.round(yc)))
                    print(refined_center)
                    center_list.append(refined_center)
                    detector_shift_x = (refined_center[0]-DetectorCenter[0])/PixelResolution
                    detector_shift_y = (refined_center[1]-DetectorCenter[1])/PixelResolution
                    shift_x_mm_list.append(detector_shift_x)
                    shift_y_mm_list.append(detector_shift_y)

                    plot_flag = True
                    results = calculate_fwhm((corrected_data, pf8_mask,(refined_center)))
                    plot_flag = False

                    ## Display plots
                    xr = first_center[0]
                    yr = first_center[1]

                    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
                    pos1 = ax1.imshow(corrected_data*mask, vmax=20, cmap="cividis")
                    ax1.scatter(xr, yr, color="blue", label=f"Detector center:({xr},{yr})")
                    ax1.scatter(
                        xc, yc, color="red", label=f"Refined center:({xc},{yc})"
                    )
                    ax1.scatter(indices[1], indices[0], facecolor="none", s=60, marker='s',edgecolor="lime", label='original peaks')
                    ax1.set_title("Center refinement: FWHM minimization")
                    fig.colorbar(pos1, ax=ax1, shrink=0.6)
                    #ax1.set_xlim(400, 2000)
                    #ax1.set_ylim(400, 2000)
                    #ax1.legend()
                    plt.savefig(f"{args.scratch}/center_refinement/plots/centered/{run_label}/{file_label}_{frame_index}.png")
                    plt.close()
                    
                
            f.close()

            f = h5py.File(f"{args.output}/centered/{run_label}/{file_label}.h5", "w")
            entry = f.create_group("entry") 
            grp_data = entry.create_group("data") 
            grp_data.create_dataset("data", data=hit_list,compression="gzip")
            grp_shots = entry.create_group("shots") 
            grp_shots.create_dataset("shift_x_mm", data=shift_x_mm_list)
            grp_shots.create_dataset("shift_y_mm", data=shift_y_mm_list)
            
            grp_config = f.create_group("beam_sweeping_config")
            for key, value in BeamSweepingParam.items():
                grp_config.create_dataset(key, data=value)
            grp_config.create_dataset('initial_center', data=initial_center_list)
            grp_config.create_dataset("refined_center", data=center_list)
            
            f.close()


if __name__ == "__main__":
    main()
