#!/usr/bin/env python3.7

from typing import List, Optional, Callable, Tuple, Any, Dict
import fabio
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import get_format, gaussian
import h5py
import math
from scipy.signal import find_peaks


DetectorCenter = [1253.5, 1157]


def main():
    parser = argparse.ArgumentParser(description="Plot calculated center distribution.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to list of data files .lst",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to output data files"
    )
    parser.add_argument(
        "-l", "--label", type=str, action="store", help="label name"
    )

    args = parser.parse_args()

    files = open(args.input, "r")
    paths = files.readlines()
    files.close()

    file_format = get_format(args.input)
    output_folder = args.output
    label = "center_sweep_" + args.label

    #print(label)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, title="Shift vertical (mm)")
    #ax2 = fig.add_subplot(122, title="Distance from last step (mm)")
    count_outliers=0
    if file_format == "lst":
        #for idx,i in enumerate(paths):
        for idx,i in enumerate(paths[:]):

            #print(idx)
            f = h5py.File(f"{i[:-1]}", "r")
            shift = np.array(f["/shift_vertical_mm"])
            #shift_cut=[x if x<8 and x>0 else (shift[pos-2]+shift[pos+2])/2 for pos,x in enumerate(shift)]
            #first_shift=min(shift_cut)
            first_shift=min(shift)
            if idx==0:
                ref_point=first_shift.copy()

            index=np.where(abs(shift-ref_point)<20)
            start=index[0][0]
                
            f.close()
            n=len(shift)
            #print(start)
            x=np.arange(-start, -start+n,1)
            ax1.plot(x, shift, label=f"File ID {i[-6:-4]}")
            #ax1.plot(x, shift_cut, label=f"File ID {i[-6:-4]}")
            """
            last_value=shift[0]
            diff=[0,]
            outlier=[]
            outlier_index=[-1]
            outlier_grad=[]

            #for idk, k in enumerate(shift_cut[1:]):
            for idk, k in enumerate(shift[1:]):

                gradient_value=(abs(k-last_value)+abs(k-shift[idk+1]))/2
                if gradient_value>0.6 and (x[idk]-outlier_index[-1])!=0:
                    print(gradient_value)                  
                    count_outliers+=1
                    outlier_grad.append(gradient_value)
                    outlier_index.append(x[idk+1])
                    outlier.append(k)
                diff.append(gradient_value)
                last_value=k
            ax2.plot(x, diff, label=f"File ID {i[-6:-4]}")
            ax2.scatter(outlier_index[1:], outlier_grad, marker='o', color='r')
            ax1.scatter(outlier_index[1:], outlier, marker='o', color='r')

            """
    ax1.set_xlabel("Frame number")
    ax1.set_xlim(0,1000)
    #ax2.set_ylim(0)
    
    ax1.legend(fontsize=8)
    #ax2.legend(fontsize=8)
    

    plt.savefig(f"{args.output}/plots/{label}.png")
    plt.show()

    print(count_outliers)


if __name__ == "__main__":
    main()
