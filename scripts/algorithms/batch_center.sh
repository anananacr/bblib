#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=0-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --nice=128
#SBATCH --mincpus=100
#SBATCH --mem=4G

#SBATCH --job-name  fwhm_center
#SBATCH --output    /gpfs/cfel/user/rodria/processed/p09/11016566/error/cc-%N-%j.out
#SBATCH --error     /gpfs/cfel/user/rodria/processed/p09/11016566/error/cc-%N-%j.err

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/p09/env-p09/bin/activate

./find_center.py -i /gpfs/cfel/user/rodria/processed/p09/11016566/lists/lyso_test.lst -m  /gpfs/cfel/user/rodria/processed/p09/11016566/lists/mask_lyso_test.lst  -c center.txt -o /gpfs/cfel/user/rodria/processed/p09/11016566/cc_4/lyso
