
:loudspeaker: :warning: :construction: Work in progress :loudspeaker: :warning: :construction:

# beambusters pipeline
Beam swepeing serial crystallography data processing routine. Scripts necessary for detector center determination based on Friedel pairs inversion symmetry. It also include auxiliary scripts for subsequent processing with CrystFEL [1].


## Python version
Python 3.10.5 (main, Jun 21 2022, 11:18:08) [GCC 4.8.5 20150623 (Red Hat 4.8.5-44)] on linux


## Usage
### Calculate the direct beam postion of each frame based on FWHM minimization of the background peak
scripts in scripts/algorithms


Raw folder: raw/run_label/scan_type/run_label_scan_type_data_*.h5
lst files: each dataset should be splitted in split_run_label_scan_type.lst**
index of lst files are integer and it will submit a slurm job for each lst index from initial_index to end_index

./turbo_center.sh split_run_label_scan_type initial_index end_index

Example:
./turbo_center.sh split_beam_sweeping_lyzo2_snake 3 10

## Contact:

Ana Carolina Rodrigues

ana.rodrigues@desy.de



