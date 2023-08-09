#!/usr/bin/env python3.7
from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import numpy as np
from utils import (
    get_format,
    mask_peaks,
    center_of_mass,
    azimuthal_average,
    gaussian,
    open_fwhm_map,
    fit_fwhm,
    shift_image_by_n_pixels,
    open_cc_map,
    get_center_theory
)
import pandas as pd
from models import PF8, PF8Info
from scipy.optimize import curve_fit
import multiprocessing
import math
import matplotlib.pyplot as plt
from scipy import signal
import h5py


def shift_and_calculate_fwhm(shift: tuple) -> Dict[str, int]:
    ## Radial average from the center of mass
    shift_x = shift[0]
    shift_y = shift[1]
    xc = center_x + shift_x
    yc = center_y + shift_y

    x, y = azimuthal_average(unbragged_data, center=(xc, yc), mask=pf8_mask)
    plt.plot(x, y)
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

def shift_and_calculate_cross_correlation(shift: tuple)-> Dict[str, float]:
    #print(shift)
    shift_x = -shift[0]
    shift_y = -shift[1]
    xc = round(data.shape[1]/2) + shift[0]
    yc = round(data.shape[0]/2) + shift[1]
    #print(xc,yc)
    shifted_data=shift_image_by_n_pixels(shift_image_by_n_pixels(data, shift_y, 0), shift_x, 1)
    shifted_mask=shift_image_by_n_pixels(shift_image_by_n_pixels(xds_mask, shift_y, 0), shift_x, 1)
    pf8_info = PF8Info(
        max_num_peaks=10000,
        pf8_detector_info=dict(
            asic_nx=shifted_mask.shape[1],
            asic_ny=shifted_mask.shape[0],
            nasics_x=1,
            nasics_y=1,
        ),
        adc_threshold=8,
        minimum_snr=3,
        min_pixel_count=1,
        max_pixel_count=200,
        local_bg_radius=3,
        min_res=0,
        max_res=100,
        _bad_pixel_map=shifted_mask,
    )

    pf8 = PF8(pf8_info)

    peak_list = pf8.get_peaks_pf8(data=shifted_data)
    
    flipped_data=shifted_data[::-1,::-1]
    pf8_info._bad_pixel_map=shifted_mask[::-1,::-1]
    pf8 = PF8(pf8_info)

    peak_list_flipped = pf8.get_peaks_pf8(data=flipped_data)
    
    if peak_list["num_peaks"]>=peak_list_flipped["num_peaks"]:
        n_peaks=peak_list_flipped["num_peaks"]
        indices = (
        np.array(peak_list["ss"][:n_peaks], dtype=int),
        np.array(peak_list["fs"][:n_peaks], dtype=int),
        )    
        indices_flipped = (
        np.array(peak_list_flipped["ss"], dtype=int),
        np.array(peak_list_flipped["fs"], dtype=int),
        )
    else:
        n_peaks=peak_list["num_peaks"]
        indices = (
        np.array(peak_list["ss"], dtype=int),
        np.array(peak_list["fs"], dtype=int),
        )    
        indices_flipped = (
        np.array(peak_list_flipped["ss"][:n_peaks], dtype=int),
        np.array(peak_list_flipped["fs"][:n_peaks], dtype=int),
        )

    #fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2,figsize=(10, 10))
    #ax1.imshow(shifted_data, vmax=10,cmap='cividis')
    #ax1.scatter(indices[1], indices[0], facecolor="none", edgecolor="red")
    #ax2.imshow(flipped_data, vmax=10, cmap='cividis')
    #ax2.scatter(indices_flipped[1], indices_flipped[0], facecolor="none", edgecolor="lime")

    
    #original image
    x_min_orig=np.min(indices[1])
    x_max_orig=np.max(indices[1])
    y_min_orig=np.min(indices[0])
    y_max_orig=np.max(indices[0])

    #flip image
    x_min_flip=np.min(indices_flipped[1])
    x_max_flip=np.max(indices_flipped[1])
    y_min_flip=np.min(indices_flipped[0])
    y_max_flip=np.max(indices_flipped[0])

    # reduced dimensions
    x_min=min(x_min_orig,x_min_flip)
    x_max=max(x_max_orig,x_max_flip)
    y_min=min(y_min_orig,y_min_flip)
    y_max=max(y_max_orig,y_max_flip)

    #print(indices)
    #print(x_min, y_min)
    img_1=np.zeros((y_max-y_min+1, x_max-x_min+1))
    x_orig=[x-x_min for x in indices[1]]
    y_orig=[x-y_min for x in indices[0]]
    img_1[y_orig, x_orig]=1
    img_1=mask_peaks(img_1, (y_orig,x_orig), 1)
    #ax3.imshow(img_1, cmap='jet')
    global mask_1
    mask_1=img_1.copy()
    mask_1[np.where(img_1==0)]=np.nan
    #print(mask_1)
    
    #print(indices_flipped)
    #print(x_min, y_min)
    img_2=np.zeros((y_max-y_min+1, x_max-x_min+1))
    x_flip=[x-x_min for x in indices_flipped[1]]
    y_flip=[x-y_min for x in indices_flipped[0]]
    img_2[y_flip, x_flip]=1
    img_2=mask_peaks(img_2, (y_flip,x_flip), 1)
    global mask_2
    mask_2=img_2.copy()
    mask_2[np.where(img_2==0)]=np.nan
    #ax4.imshow(img_2, cmap='jet')
    #plt.show()
    img_2[np.where(img_2==0)]=np.nan
    cc_matrix=correlate_2d(img_1,img_2, mask_1, mask_2)
    #plt.imshow(cc_matrix, vmax=1, cmap='jet')
    row, col=cc_matrix.shape
    row=round(row/4)
    col=round(col/4)
    reduced_cc_matrix=cc_matrix[row:-row, col:-col]

    row, col=reduced_cc_matrix.shape
    row=round(row/2)
    col=round(col/2)
    sub_reduced_cc_matrix=reduced_cc_matrix[row-10:row+10,col-10:col+10]

    maximum_index=np.where(sub_reduced_cc_matrix==np.max(sub_reduced_cc_matrix))
    non_zero_index=np.where(sub_reduced_cc_matrix!=0)
    index=np.unravel_index(np.argmax(np.abs(sub_reduced_cc_matrix)), sub_reduced_cc_matrix.shape)
    
    xx, yy = np.meshgrid(np.arange(-img_1.shape[1]/2,img_1.shape[1]/2, 1, dtype=int), np.arange(-img_1.shape[0]/2,img_1.shape[0]/2, 1, dtype=int))
    xx=xx[row-10:row+10,col-10:col+10]
    yy=yy[row-10:row+10,col-10:col+10]
    orig_xc=xc
    orig_yc=yc

    xc+=(xx[index])/2
    yc+=(yy[index])/2

    max_candidates=[]
    for index in zip(*maximum_index):
        max_candidates.append([orig_xc+((xx[index])/2), orig_yc+((yy[index])/2)])

    non_zero_candidates=[]
    for index in zip(*non_zero_index):
        non_zero_candidates.append([orig_xc+((xx[index])/2), orig_yc+((yy[index])/2)])
    print('Refined center',xc,yc)
    return {
        "max_index": maximum_index,
        "non_zero_index": non_zero_index,
        "index": index,
        "xc":xc,
        "yc": yc,
        "cc_matrix": cc_matrix,
        "reduced_cc_matrix": reduced_cc_matrix,
        "sub_reduced_cc_matrix": sub_reduced_cc_matrix,
        "xx": xx,
        "yy": yy,
        "max_candidates": max_candidates,
        "non_zero_candidates": non_zero_candidates
    }
    
def calculate_product(shift:Tuple[int])->float:
    im1=mask_1
    im2=mask_2
    shift_x=shift[0]
    shift_y=shift[1]

    im2=shift_image_by_n_pixels(shift_image_by_n_pixels(im2, shift_y, 0), shift_x, 1)
    im2[np.where(im2==0)]=np.nan
    cc=0
    for idy, j in enumerate(im1):
        for idx, i in enumerate(j):
            if not np.isnan(i) and not np.isnan(im2[idy,idx]):
               cc+=i*im2[idy,idx]
    return cc

def correlate_2d(im1:np.ndarray, im2:np.ndarray, mask_1:np.ndarray, mask_2:np.ndarray)->np.ndarray:
    #print(im1.shape)
    corr=np.ndarray((im1.shape[0]+im2.shape[0], (im1.shape[1]+im2.shape[1])))
    xx, yy = np.meshgrid(np.arange(-im1.shape[1],im1.shape[1], 1, dtype=int), np.arange(-im1.shape[0],im1.shape[0], 1, dtype=int))
    coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))

    """
    for idy, j in enumerate(corr):
        for idx, i in enumerate(j):
            #print(xx[idy,idx], yy[idy,idx])
            corr[idy, idx]=calculate_product(mask_1, mask_2, xx[idy,idx], yy[idy,idx])
            #print(corr[idy, idx])
    """
    
    pool = multiprocessing.Pool()
    with pool:
        cc_summary = pool.map(calculate_product, coordinates)

    corr = np.array(cc_summary).reshape((corr.shape)) 

    return corr

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
        "-center",
        "--center",
        type=str,
        action="store",
        help="path to list of theoretical center positions file in .txt",
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
    table_real_center, loaded_table = get_center_theory(paths, args.center)
    #(table_real_center)
    if file_format == "lst":
        ref_image = []
        for i in range(0, len(paths[:])):
        #for i in range(0, 1):
            file_name = paths[i][:-1]
            print(file_name)
            if get_format(file_name) == "cbf":
                global data
                data = np.array(fabio.open(f"{file_name}").data)
                mask_file_name = mask_paths[i][:-1]

                global xds_mask
                xds_mask = np.array(fabio.open(f"{mask_file_name}").data)
                # Mask of defective pixels
                xds_mask[np.where(xds_mask <= 0)] = 0
                xds_mask[np.where(xds_mask > 0)] = 1
                # Mask hot pixels
                xds_mask[np.where(data > 1e3)] = 0

                real_center = table_real_center[i]
                
                """
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

                first_xc = xc
                first_yc = yc

                global center_x
                center_x = xc
                global center_y
                center_y = yc
                print("First approximation", xc, yc)

                ## Display first approximation plots
                
                xr=831
                yr=833
                pos=plt.imshow(unbragged_data, vmax=7, cmap='jet')
                plt.scatter(xr,yr, color='lime', label='xds')
                plt.scatter(xc,yc, color='r', label='center of mass')
                plt.title('First approximation: center of mass')
                plt.colorbar(pos,shrink=0.6)
                plt.legend()
                plt.savefig(f'/home/rodria/Desktop/com/lyso_{i}_error_x_{xc-xr}_y_{yc-yr}.png')
                plt.show()
                
                ## Grid search of sharpness of the azimutal average
                xx, yy = np.meshgrid(
                    np.arange(-30, 31, 1, dtype=int), np.arange(-30, 31, 1, dtype=int)
                )
                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))

                pool = multiprocessing.Pool()
                with pool:
                    fwhm_summary = pool.map(shift_and_calculate_fwhm, coordinates)

                ## Display plots
                # open_fwhm_map(fwhm_summary, i)

                ## Second aproximation of the direct beam

                xc, yc = fit_fwhm(fwhm_summary)
                print("Second approximation", xc, yc)

                ## Display plots
                
                xr=831
                yr=761
                fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))
                pos1=ax1.imshow(unbragged_data, vmax=7, cmap='jet')
                ax1.scatter(xr,yr, color='lime', label='xds')
                ax1.scatter(first_xc,first_yc, color='r', label='calculated center')
                ax1.set_title('First approximation: center of mass')
                fig.colorbar(pos1, ax=ax1,shrink=0.6)
                ax1.legend()

                pos2=ax2.imshow(unbragged_data, vmax=7, cmap='jet')
                ax2.scatter(xr,yr, color='lime', label='xds')
                ax2.scatter(xc,yc, color='blueviolet', label='calculated center')
                ax2.set_title('Second approximation: FWHM/R minimization')
                fig.colorbar(pos2, ax=ax2,shrink=0.6)
                ax2.legend()
                plt.savefig(f'/home/rodria/Desktop/second/lyso_{i}.png')
                plt.show()
                """
                ## Check pairs of Friedel
                #print(real_center)
                x_min=-(data.shape[1]/2)+real_center[0]-0
                x_max=-(data.shape[1]/2)+real_center[0]+1+0
                y_min=-(data.shape[0]/2)+real_center[1]-0
                y_max=-(data.shape[0]/2)+real_center[1]+1+0
                #print(x_min,x_max, y_min,y_max)
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, 1, dtype=int), np.arange(y_min, y_max, 1, dtype=int)
                )
                coordinates = np.column_stack((np.ravel(xx), np.ravel(yy)))
                results=shift_and_calculate_cross_correlation(coordinates[0])
                f = h5py.File(f"{args.output}_{i}.h5", "w")
                for key in results:
                    f.create_dataset(key, data=results[key])

                f.close()
if __name__ == "__main__":
    main()
