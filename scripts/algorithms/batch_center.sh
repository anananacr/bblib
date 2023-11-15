#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=1-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mincpus=128
#SBATCH --mem=40G

#SBATCH --job-name  fwhm_center
#SBATCH --output    /asap3/petra3/gpfs/p09/2023/data/11019087/scratch_cc/rodria/error/cc-%N-%j.out
#SBATCH --error     /asap3/petra3/gpfs/p09/2023/data/11019087/scratch_cc/rodria/error/cc-%N-%j.err

INPUT=$1
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019087/processed/rodria
SCRATCH=/asap3/petra3/gpfs/p09/2023/data/11019087/scratch_cc/rodria

source /etc/profile.d/modules.sh
module load maxwell python/3.6
source /home/rodria/scripts/p09/env-p09/bin/activate

python find_center.py -i ${SCRATCH}/lists/${INPUT}.lst -o ${ROOT} -s ${SCRATCH} -m /asap3/petra3/gpfs/p09/2023/data/11019087/scratch_cc/rodria/masks/mask_ana_v2.h5 -g /asap3/petra3/gpfs/p09/2023/data/11019087/processed/galchenm/beam_sweeping/powder_direct_beam/eiger500k_corrected_beam_centre.geom;
