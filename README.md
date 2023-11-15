# p09-utils
Python scripts for data processing at P09 - PETRA III  (DESY).

Using Python 3.7.13

:loudspeaker: :warning: :construction: Work in progress :loudspeaker: :warning: :construction:

## Beam sweeping

### Calculate the direct beam postion of each frame based on FWHM minimization of the background peak

Raw folder: raw/run_label/scan_type/run_label_scan_type_data_*.h5
lst files: each dataset should be splitted in split_run_label_scan_type.lst**
index of lst files are integer and it will submit a slurm job for each lst index from initial_index to end_index

./turbo_center.sh split_run_label_scan_type initial_index end_index

Example:
./turbo_center.sh split_beam_sweeping_lyzo2_snake 3 10

Contact:

Ana Carolina Rodrigues

ana.rodrigues@desy.de



