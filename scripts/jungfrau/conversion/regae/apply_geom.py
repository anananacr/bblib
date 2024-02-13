import h5py
import argparse
import numpy as np
import om.utils.crystfel_geometry as crystfel_geometry


def apply_geom(data: np.ndarray, geometry_filename: str) -> np.ndarray:
    ## Apply crystfel geomtry file .geom
    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(geometry_filename)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2 * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min()))) + 2
    )
    x_minimum: int = (
        2 * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min()))) + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)
    _img_center_x: int = int(visual_img_shape[1] / 2)
    _img_center_y: int = int(visual_img_shape[0] / 2)

    corr_data = crystfel_geometry.apply_geometry_to_data(data, geometry)
    return corr_data


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Convert JUNGFRAU 1M H5 images collected at REGAE for rotational data step/fly scan and return images in rotation sequence according tro the file index."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )

    parser.add_argument(
        "-s", "--start_index", type=int, action="store", help="starting file index"
    )
    parser.add_argument(
        "-e", "--end_index", type=int, action="store", help="ending file index"
    )
    parser.add_argument(
        "-f",
        "--frames",
        default=None,
        type=int,
        action="store",
        help="If more than one frame was measured per step. Number of frames to be accumulated per step for rotational step manner. None for fly scan.",
    )
    parser.add_argument(
        "-g", "--geom", type=str, action="store", help="crystfel geometry file"
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="hdf5 output path"
    )
    args = parser.parse_args(raw_args)

    index = np.arange(args.start_index, args.end_index, 1)
    n_frames = args.end_index - args.start_index

    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(args.geom)
    _pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    y_minimum: int = (
        2 * int(max(abs(_pixelmaps["y"].max()), abs(_pixelmaps["y"].min()))) + 2
    )
    x_minimum: int = (
        2 * int(max(abs(_pixelmaps["x"].max()), abs(_pixelmaps["x"].min()))) + 2
    )
    visual_img_shape: Tuple[int, int] = (y_minimum, x_minimum)

    for idx, i in enumerate(index):
        f = h5py.File(f"{args.input}_master_{i}.h5", "r")

        size = len(f["data"])

        try:
            raw = np.array(f["data"][:size])
        except OSError:
            print("skipped", i)
            continue

        if args.frames == None:
            n_frames_measured = raw.shape[0]
        else:
            n_frames_measured = args.frames

        corr_frame = np.zeros(
            (n_frames_measured, visual_img_shape[0], visual_img_shape[1]),
            dtype=np.int32,
        )
        f.close()

        for idy, j in enumerate(raw[:n_frames_measured]):
            corr_frame[idy] = apply_geom(j, args.geom)

    g = h5py.File(args.output + ".h5", "w")
    g.create_dataset("data", data=corr_frame, compression="gzip")
    g.close()


if __name__ == "__main__":
    main()
