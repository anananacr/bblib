#!/bin/sh

#SBATCH --partition=allcpu,upex
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mincpus=32
#SBATCH --mem=10G
#SBATCH --nice=100
#SBATCH --job-name  bb
#SBATCH --output   /asap3/fs-bmx/gpfs/regae/2022/data/11016614/processed/test/rodria/error/bb-%N-%j.out
#SBATCH --error    /asap3/fs-bmx/gpfs/regae/2022/data/11016614/processed/test/rodria/error/bb-%N-%j.err

INPUT=$1
ROOT=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/processed/test/rodria
SCRATCH=/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/test

source /etc/profile.d/modules.sh
source /home/rodria/scripts/p09/env-p09/bin/activate

python find_center.py -i ${ROOT}/lists/${INPUT} -o ${ROOT}/.. -s ${SCRATCH} -g ${ROOT}/geoms/JF_regae_v4_altered.geom
