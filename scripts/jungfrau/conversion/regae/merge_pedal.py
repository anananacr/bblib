import h5py
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Average pedestals.")
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        action="store",
        help="raw files label 231020_mica_c4_m1_001_001",
    )

    args = parser.parse_args()
    label = args.label
    root = "/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/darks"
    f = h5py.File(f"{root}/pedal_d0_{label}_start.h5", "r")
    g = h5py.File(f"{root}/pedal_d0_{label}_stop.h5", "r")
    output = h5py.File(f"{root}/pedal_d0_{label}_average.h5", "w")
    for key in f.keys():
        data = (np.array(f[key]) + np.array(g[key])) // 2
        output.create_dataset(key, data=data)
    f.close()
    g.close()
    output.close()

    f = h5py.File(f"{root}/pedal_d1_{label}_start.h5", "r")
    g = h5py.File(f"{root}/pedal_d1_{label}_stop.h5", "r")
    output = h5py.File(f"{root}/pedal_d1_{label}_average.h5", "w")
    for key in f.keys():
        data = (np.array(f[key]) + np.array(g[key])) // 2
        output.create_dataset(key, data=data)
    f.close()
    g.close()
    output.close()


if __name__ == "__main__":
    main()
