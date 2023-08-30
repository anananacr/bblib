#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=0-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --nice=128
#SBATCH --mincpus=100
#SBATCH --mem=4G

#SBATCH --job-name  f2x_center
#SBATCH --output    /gpfs/cfel/user/rodria/processed/p09/11016750/error/center-%N-%j.out
#SBATCH --error     /gpfs/cfel/user/rodria/processed/p09/11016750/error/center-%N-%j.err

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/p09/env-p09/bin/activate
cp /gpfs/cfel/user/rodria/processed/p09/11016750/geom/pilatus6M_219mm.geom /asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria/geom/f2x_04_03
./find_center.py -i /gpfs/cfel/user/rodria/processed/p09/11016750/lists/f2x_chip_04_fly_03.lst -m  /gpfs/cfel/user/rodria/processed/p09/11016750/lists/mask_f2x.lst -o /asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria/f2x_04_03/f2x_04_03 -g /asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria/geom/f2x_04_03/pilatus6M_219mm.geom;
