import h5py
import argparse
import numpy as np
import om.lib.geometry as geometry
import os
import subprocess as sub
from PIL import Image
import fabio

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
    geometry_txt = open(args.geom, "r").readlines()

    geom = geometry.GeometryInformation(
        geometry_description=geometry_txt, geometry_format="crystfel"
    )
    mask_file = [
        x.split(" = ")[-1][:-1]
        for x in geometry_txt
        if x.split(" = ")[0] == "mask_file"
    ][0]
    mask_path = [
        x.split(" = ")[-1][:-1] for x in geometry_txt if x.split(" = ")[0] == "mask"
    ][0]

    pixel_maps = geom.get_pixel_maps()
    detector_layout = geom.get_layout_info()

    y_minimum: int = (
        2 * int(max(abs(pixel_maps["y"].max()), abs(pixel_maps["y"].min()))) + 2
    )
    x_minimum: int = (
        2 * int(max(abs(pixel_maps["x"].max()), abs(pixel_maps["x"].min()))) + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)

    data_visualize = geometry.DataVisualizer(pixel_maps=pixel_maps)

    raw_folder = os.path.dirname(args.input)
    output_folder = args.output

    if not os.path.isdir(output_folder):
        cmd = f"mkdir {output_folder}"
        sub.call(cmd, shell=True)

    if os.path.isfile(f"{raw_folder}/info.txt"):
        cmd = f"cp {raw_folder}/info.txt {output_folder}"
        sub.call(cmd, shell=True)

    with h5py.File(f"{mask_file}", "r") as f:
        mask = np.array(f[f"{mask_path}"])

    f = h5py.File(f"{args.input}", "r")
    size = len(f["data"])

    label = ((args.input).split("/")[-1]).split('.')[0]

    for i in range(size):
        try:
            raw = np.array(f["data"][i])
            raw[np.where(raw <= 0)] = -1
        except OSError:
            print("skipped", i)
            continue
        corr_frame = np.zeros(
            (visual_img_shape[0], visual_img_shape[1]), dtype=np.int16
        )
        corr_frame = data_visualize.visualize_data(data=raw*mask)
        corr_frame[np.where(corr_frame <= 0)] = -1
        output_filename=f"{args.output}/{label[:-7]}_{i:06}"
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
