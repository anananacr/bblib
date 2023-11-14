#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=0-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mincpus=4
#SBATCH --mem=40G

#SBATCH --job-name  fwhm_center
#SBATCH --output    /asap3/petra3/gpfs/p09/2023/data/11019087/scratch_cc/rodria/error/cc-%N-%j.out
#SBATCH --error     /asap3/petra3/gpfs/p09/2023/data/11019087/scratch_cc/rodria/error/cc-%N-%j.err

INPUT=$1
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019087/processed/rodria
SCRATCH=/asap3/petra3/gpfs/p09/2023/data/11019087/scratch_cc/rodria

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/p09/env-p09/bin/activate

./find_center.py -i ${SCRATCH}/lists/${INPUT}.lst -o ${ROOT} -s ${SCRATCH} -g /asap3/petra3/gpfs/p09/2023/data/11019087/processed/galchenm/beam_sweeping/geoms/eiger500k.geom;
