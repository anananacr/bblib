#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --nice=0
#SBATCH --mincpus=48
#SBATCH --mem=4G

#SBATCH --job-name  f2x_center
#SBATCH --output    /gpfs/cfel/user/rodria/processed/p09/11016750/error/center-%N-%j.out
#SBATCH --error     /gpfs/cfel/user/rodria/processed/p09/11016750/error/center-%N-%j.err

ROOT=/asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria
source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/p09/env-p09/bin/activate
cp /gpfs/cfel/user/rodria/processed/p09/11016750/geom/pilatus6M_219mm.geom ${ROOT}/geom/f2x_D_01_01
./find_center.py -i ${ROOT}/f2x_D_01_01/lists/split_f2x_D_01_01.lst10 -m  ${ROOT}/f2x_D_01_01/lists/mask_f2x.lst -o ${ROOT}/f2x_D_01_01 -g ${ROOT}/geom/f2x_D_01_01/pilatus6M_219mm.geom;