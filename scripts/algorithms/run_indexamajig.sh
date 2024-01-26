#!/bin/sh
source /etc/profile.d/modules.sh
module purge
module load maxwell crystfel/0.10.2
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria
INPUT=$1
OUTPUT=$2
CENTER=$3
INDEX=$4

rm /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/streams/${OUTPUT}.stream
if [ "$CENTER" -eq 0 ];
then
    command="indexamajig -i ${ROOT}/lists/${INPUT}.lst -o  ${ROOT}/streams/${OUTPUT}.stream -g ${ROOT}/geoms/eiger500k_corrected_beam_centre.geom --peaks=peakfinder8"
else 
    command="indexamajig -i ${ROOT}/lists/${INPUT}.lst -o  ${ROOT}/streams/${OUTPUT}.stream -g ${ROOT}/geoms/eiger500k_corrected_beam_centre_shift_fosakp.geom --peaks=peakfinder8"
fi

if [ "$INDEX" -eq 0 ];
then
    command="$command --indexing=none"
fi

#refine pf8 2
#test_0
#command="$command -j 32 --threshold=20 --min-snr=9 --min-pix-count=3 --max-pix-count=20 --max-res=330 --int-radius=3,4,5;"
#test_1
command="$command -j 32 --threshold=40 --min-snr=5 --min-pix-count=1 --max-pix-count=200 --max-res=1200 --min-peaks=10 --int-radius=3,4,5;"

$command
