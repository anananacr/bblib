#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=1-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mincpus=128
#SBATCH --mem=10G

#SBATCH --job-name  write_peaks_lyso_index
#SBATCH --output    /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/cc-%N-%j.out
#SBATCH --error     /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/cc-%N-%j.err

source /etc/profile.d/modules.sh
source /home/rodria/scripts/p09/env-p09/bin/activate
python write_original_peaks_list_for_integration.py lyso_indexing/lyso_agg_hits