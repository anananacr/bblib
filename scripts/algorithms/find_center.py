#!/usr/bin/env python3.7
from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import numpy as np
from utils import get_format, mask_peaks, center_of_mass, azimuthal_average, gaussian
from models import PF8, PF8Info
from scipy.optimize import curve_fit
import multiprocessing
import math
import matplotlib.pyplot as plt

def shift_and_calculate_fwhm(shift: tuple) -> Dict[str, int]:
    ## Radial average from the center of mass
    shift_x = shift[0]
    shift_y = shift[1]
    xc = center_x + shift_x
    yc = center_y + shift_y

    x, y = azimuthal_average(unbragged_data, center=(xc, yc), bad_px_mask=pf8_mask)
    plt.plot(x,y)
    ## Define background peak region
    x_min = 150
    x_max = 400
    x = x[x_min:x_max]
    y = y[x_min:x_max]

    ## Estimation of initial parameters
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
    residuals = y - gaussian(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    ## Calculation of FWHM
    fwhm = popt[2] * math.sqrt(8 * np.log(2))

    ## Divide by radius of the peak to get shasrpness ratio
    fwhm_over_radius = fwhm / popt[1]

    ## Display plots
    """
    x_fit=x.copy()
    y_fit=gaussian(x_fit, *popt)

    plt.plot(x,y)
    plt.plot(x_fit,y_fit, 'r:', label=f'gaussian fit \n a:{round(popt[0],2)} \n x0:{round(popt[1],2)} \n sigma:{round(popt[2],2)} \n R² {round(r_squared, 4)}\n FWHM/R : {round(fwhm_over_radius,3)}')
    plt.title('Azimuthal integration')
    plt.legend()
    plt.savefig(f'/home/rodria/Desktop/radial/lyso_shift_{shift[0]}_{shift[1]}.png')
    plt.show()
    """
    return {
        "shift_x": shift_x,
        "shift_y": shift_y,
        "xc": xc,
        "yc": yc,
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
        "-o", "--output", type=str, action="store", help="path to output data files"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    mask_files = open(args.mask, "r")
    mask_paths = mask_files.readlines()
    mask_files.close()

    file_format = get_format(args.input)
    if file_format == "lst":
        ref_image = []
        for i in range(0,len(paths[:])):
            file_name = paths[i][:-1]
            print(file_name)
            if get_format(file_name) == "cbf":
                data = np.array(fabio.open(f"{file_name}").data)
                mask_file_name = mask_paths[i][:-1]
                xds_mask = np.array(fabio.open(f"{mask_file_name}").data)
                # Mask of defective pixels
                xds_mask[np.where(xds_mask <= 0)] = 0
                xds_mask[np.where(xds_mask > 0)] = 1
                # Mask hot pixels
                xds_mask[np.where(data > 1e3)] = 0

                ## Find peaks with peakfinder8 and mask peaks
                pf8_info = PF8Info(
                    max_num_peaks=10000,
                    pf8_detector_info=dict(
                        asic_nx=xds_mask.shape[1],
                        asic_ny=xds_mask.shape[0],
                        nasics_x=1,
                        nasics_y=1,
                    ),
                    adc_threshold=10,
                    minimum_snr=5,
                    min_pixel_count=1,
                    max_pixel_count=200,
                    local_bg_radius=3,
                    min_res=0,
                    max_res=10000,
                    _bad_pixel_map=xds_mask,
                )

                pf8 = PF8(pf8_info)

                peak_list = pf8.get_peaks_pf8(data=data)
                indices = (
                    np.array(peak_list["ss"], dtype=int),
                    np.array(peak_list["fs"], dtype=int),
                )
                # Mask Bragg  peaks
                mask = xds_mask.copy()
                only_peaks_mask = mask_peaks(np.ones_like(mask), indices, bragg=0)
                xds_and_peaks_mask = mask_peaks(mask, indices, bragg=0)
                global pf8_mask
                pf8_mask = xds_and_peaks_mask
                global unbragged_data
                unbragged_data = data * pf8_mask

                ## Approximate center of mass
                xc, yc = center_of_mass(unbragged_data)

                ## Center of mass again with the flipped image to account for eventual background asymmetry

                flipped_data = unbragged_data[::-1, ::-1]
                xc_flip, yc_flip = center_of_mass(flipped_data)

                h, w = data.shape
                shift_x = w / 2 - xc
                shift_y = h / 2 - yc
                shift_x_flip = w / 2 - xc_flip
                shift_y_flip = h / 2 - yc_flip

                diff_x = abs((abs(shift_x) - abs(shift_x_flip)) / 2)
                diff_y = abs((abs(shift_y) - abs(shift_y_flip)) / 2)

                if shift_x <= 0:
                    shift_x -= diff_x
                else:
                    shift_x += diff_x
                if shift_y <= 0:
                    shift_y -= diff_y
                else:
                    shift_y += diff_y
                ## First approximation of the direct beam
                xc = int(round(w / 2 - shift_x))
                yc = int(round(h / 2 - shift_y))

                global center_x
                center_x = xc
                global center_y
                center_y = yc
                print(xc,yc)
                
                ## Display first approximation plots
                """
                xr=831
                yr=833
                plt.imshow(unbragged_data, vmax=10, cmap='jet')
                plt.scatter(xr,yr, color='lime', label='xds')
                plt.scatter(xc,yc, color='r', label='center of mass')
                plt.title('First approximation: center of mass')
                plt.legend()
                plt.savefig(f'/home/rodria/Desktop/com/lyso_25_error_{xc-xr}_{yc-yr}.png')
                plt.show()

                """
                ## Grid search of sharpness of the azimutal average
                xx, yy = np.meshgrid(
                    np.arange(-30, 31, 1, dtype=int), np.arange(-30, 31, 1, dtype=int)
                )
                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))

                pool = multiprocessing.Pool()
                with pool:
                    result = pool.map(shift_and_calculate_fwhm, coordinates)
                

                f = open(f"{args.output}_{i}.txt", "w")
                for j in result:
                    f.write(f"{j}\n")
                f.close()

                ## Second aproximation of the direct beam

                ## Check for pairs of Friedel in the image
                

if __name__ == "__main__":
    main()
