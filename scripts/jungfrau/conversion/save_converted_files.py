import h5py
import argparse
import numpy as np
import om.utils.crystfel_geometry as crystfel_geometry
import fabio
import os
import subprocess as sub
from PIL import Image


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Convert JUNGFRAU 1M H5 images collected at REGAE for rotational data step/fly scan and return images in rotation sequence according tro the file index."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )

    parser.add_argument(
        "-g", "--geom", type=str, action="store", help="crystfel geometry file"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="hdf5 output path"
    )
    parser.add_argument(
        "-f", "--format", type=str, action="store", help="output format tif or cbf"
    )
    args = parser.parse_args(raw_args)

    raw_folder = os.path.dirname(args.input)
    output_folder = args.output
    if not os.path.isdir(output_folder):
        cmd = f"mkdir {output_folder}"
        sub.call(cmd, shell=True)

    if os.path.isfile(f"{raw_folder}/info.txt"):
        cmd = f"cp {raw_folder}/info.txt {output_folder}"
        sub.call(cmd, shell=True)

    f = h5py.File(f"{args.input}", "r")
    size = len(f["data"])

    label = (args.input).split("/")[-1]

    for i in range(size):
        try:
            raw = np.array(f["data"][i])
            raw[np.where(raw <= 0)] = -1
        except OSError:
            print("skipped", i)
            continue
        corr_frame = np.zeros(
            (1024,1024), dtype=np.int32
        )
        corr_frame = raw
        corr_frame[np.where(corr_frame <= 0)] = -1
        
        output_filename=f"{args.output}/{label}_{i:06}"

        if args.format=='tif':
            Image.fromarray(corr_frame).save(output_filename+".tif")
        elif args.format=='cbf':
            cbf = fabio.cbfimage.cbfimage(data=corr_frame.astype(np.int16))
            cbf.write(output_filename + ".cbf")
        else:
            print("Output not recognized")
            f.close()
            break

    f.close()


if __name__ == "__main__":
    main()