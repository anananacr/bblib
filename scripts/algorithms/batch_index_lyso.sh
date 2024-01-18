#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=1-23:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mincpus=128
#SBATCH --mem=10G

#SBATCH --job-name  index
#SBATCH --output    /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/indexamajig-%N-%j.out
#SBATCH --error     /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/indexamajig-%N-%j.err

INPUT=$1
OUTPUT=$2
CELL=$3


ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria

source /etc/profile.d/modules.sh
module purge
module load maxwell crystfel/0.10.2

rm /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/streams/${OUTPUT}.stream
command="indexamajig -i ${ROOT}/lists/${INPUT}.lst -o  ${ROOT}/streams/${OUTPUT}.stream -g ${ROOT}/geoms/eiger500k_corrected_beam_centre_lyso_for_index.geom  --peaks=cxi "
if [ -n "$CELL" ];
then
command="$command -p ${ROOT}/cell/${CELL}"
fi
command="$command -j 128"
echo $command
$command