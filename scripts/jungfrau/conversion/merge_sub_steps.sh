#!/bin/sh

#SBATCH --partition=allcpu,upex
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=40G

#SBATCH --job-name  merge_frames
#SBATCH --output    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/center_refinement/error/merge-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/center_refinement/error/merge-%N-%j.err


# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in step-wise manner, using start and end frame number arguments,  dividing it in smaller jobs.

# ./convert_step.sh my-file.h5 label

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module purge all
source /home/rodria/scripts/regae/env-regae/bin/activate

ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/processed/test/converted
INPUT=$1
N=$2

python merge_sub_steps.py -i ${ROOT}/${INPUT} -n ${N} ;

