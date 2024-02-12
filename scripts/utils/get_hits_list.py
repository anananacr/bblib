import sys
import numpy as np
import h5py
from os.path import basename, splitext

# from utils import shift_image_by_n_pixels
if sys.argv[1] == "-":
    stream = sys.stdin
else:
    stream = open(
        "/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/streams/"
        + sys.argv[1]
        + ".stream",
        "r",
    )


reading_geometry = False
reading_chunks = False
reading_peaks = False
is_a_hit = False
max_fs = -1
max_ss = -1
print(splitext(basename(sys.argv[1]))[0])
files_lst = open(
    f"/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/lists/{splitext(basename(sys.argv[1]))[0]}all_merged_before_indexing-hits.lst",
    "w",
)

for count, line in enumerate(stream):
    if reading_chunks:
        if line.startswith("End of peak list"):
            reading_peaks = False
        elif line.startswith("  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel"):
            reading_peaks = True
        elif line.split(": ")[0] == "Image filename":
            file_name = line.split(": ")[-1][:-1]
        elif line.split(": ")[0] == "Event":
            event = int(line.split(": //")[-1])
        elif line.startswith("hit = 1"):
            is_a_hit = True
            files_lst.write(f"{file_name} //{event}\n")
        elif line.startswith("hit = 0"):
            is_a_hit = False
    elif line.startswith("----- End geometry file -----"):
        reading_geometry = False
    elif reading_geometry:
        try:
            par, val = line.split("=")
            if par.split("/")[-1].strip() == "max_fs" and int(val) > max_fs:
                max_fs = int(val)
            elif par.split("/")[-1].strip() == "max_ss" and int(val) > max_ss:
                max_ss = int(val)
        except ValueError:
            pass
    elif line.startswith("----- Begin geometry file -----"):
        reading_geometry = True
    elif line.startswith("----- Begin chunk -----"):
        reading_chunks = True
files_lst.close()
