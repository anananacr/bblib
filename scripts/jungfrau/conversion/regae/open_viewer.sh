#!/bin/sh
## Written by: Ana Carolina Rodrigues (ana.rodrigues@desy.de)
## How to run: open_viewer.sh

beamtime=11018148
ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/$beamtime
source /etc/profile.d/modules.sh
module purge

echo "----------------------- Opening images folder -----------------------"
module load xray;

adxv ${ROOT}/processed/assembled/