#!/bin/sh


#SBATCH --partition=allcpu,cfel
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --nice=0
#SBATCH --mincpus=128
#SBATCH --mem=20G

#SBATCH --job-name  f2x_hit
#SBATCH --output    /asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria/error/center-%N-%j.out
#SBATCH --error     /asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria/error/center-%N-%j.err
INPUT=f2x_D_01_01
ROOT=/asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria
source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/p09/env-p09/bin/activate

./hit_pre_filter.py -i ${ROOT}/${INPUT}/lists/${INPUT}.lst -m  ${ROOT}/${INPUT}/lists/mask_f2x.lst -o ${ROOT}/${INPUT};
