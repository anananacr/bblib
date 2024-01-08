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
CENTER=$3
INDEX=$4
DIST=$5

ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria

source /etc/profile.d/modules.sh
module purge
module load maxwell crystfel/0.10.2

rm /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/streams/${OUTPUT}.stream
if [ "$CENTER" -eq 0 ];
then
    command="indexamajig -i ${ROOT}/lists/${INPUT}.lst -o  ${ROOT}/streams/${OUTPUT}.stream -g ${ROOT}/geoms/eiger500k_corrected_beam_centre.geom --peaks=peakfinder8"
else
    if [ "$DIST" -gt 0 ];
    then
    command="indexamajig -i ${ROOT}/lists/${INPUT}.lst -o  ${ROOT}/streams/${OUTPUT}_${DIST}.stream -g ${ROOT}/geoms/eiger500k_corrected_beam_centre_shift_${DIST}.geom --peaks=peakfinder8"
    else
    command="indexamajig -i ${ROOT}/lists/${INPUT}.lst -o  ${ROOT}/streams/${OUTPUT}.stream -g ${ROOT}/geoms/eiger500k_corrected_beam_centre_shift.geom --peaks=peakfinder8"
    fi
fi

if [ "$INDEX" -eq 0 ];
then
    command="$command --indexing=none"
fi



command="$command -j 128 --threshold=20 --min-snr=5 --min-pix-count=2 --max-pix-count=10 --int-radius=3,4,5;"
#command="$command -j 128 --threshold=40 --min-snr=5 --min-pix-count=2 --max-pix-count=10 --int-radius=3,4,5;"
echo $command
$command