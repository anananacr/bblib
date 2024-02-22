#!/bin/sh
source /etc/profile.d/modules.sh
module purge
module load maxwell crystfel/0.10.2
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria
INPUT=$1
OUTPUT=$2

rm /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/streams/${OUTPUT}.stream

command="indexamajig -i ${ROOT}/lists/${INPUT}.lst -o  ${ROOT}/streams/${OUTPUT}.stream -g ${ROOT}/geoms/eiger500k_corrected_beam_centre_shift_fosakp.geom --peaks=peakfinder8 --indexing=none"

command="$command -j 64 --threshold=30 --min-snr=3.5 --min-pix-count=1 --max-pix-count=200 --min-res=0 --max-res=1200 --min-peaks=10 --int-radius=3,4,5 --copy-header=/shift_vertical_mm --copy-header=/shift_horizonthal_mm"

$command
