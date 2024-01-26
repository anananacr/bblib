#!/bin/sh
LABEL=$1
OUTPUT=$2
OUTPUT_FOLDER=$(dirname "$LABEL")
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria

cp ${ROOT}/streams/${LABEL}_0.stream ${ROOT}/streams/${OUTPUT_FOLDER}/${OUTPUT}
for i in $(seq 1 1 26); do
    cat ${ROOT}/streams/${LABEL}_${i}.stream >> ${ROOT}/streams/${OUTPUT_FOLDER}/${OUTPUT}
done