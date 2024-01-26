#!/bin/sh
LABEL=$1
OUTPUT=$2
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria

cp ${ROOT}/lists/${LABEL}_0-hits.lst ${ROOT}/lists/${OUTPUT}
for i in $(seq 1 1 26); do
    cat ${ROOT}/lists/${LABEL}_${i}-hits.lst >> ${ROOT}/lists/${OUTPUT}
done