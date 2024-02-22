#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=1-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mincpus=64
#SBATCH --mem=20G

#SBATCH --job-name  write_peaks_lyso_index
#SBATCH --output    /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/cc-%N-%j.out
#SBATCH --error     /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/cc-%N-%j.err

source /etc/profile.d/modules.sh
source /home/rodria/scripts/p09/env-p09/bin/activate
#python write_peak_list_for_indexing.py lyso_powder_new/lyso_centered_new
python write_peak_list_for_indexing.py fakp_powder_new/fakp_new