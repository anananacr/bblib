#!/bin/sh
## Written by: Ana Carolina Rodrigues (ana.rodrigues@desy.de)
## How to run: open_screening.sh folder_onraw/ed_screening_001/file_label



SCRIPTS_FOLDER=/home/rodria/scripts/regae/screen_scripts
beamtime=11018148
INP=$1

if [ -z "$1" ]; then
    echo "Error: Missing the first argument."
    echo "Usage: $0 <folder_on_raw/ed_screening_00*/folder_on_raw_screen_00*> "
    echo "Example: $0 <231207_test_screen/ed_screening_002/231207_test_screen_002> "

    exit 1
fi

echo "Convert images for data: $1"

MODE=screening
N=$(basename $INP)
START=0
END=0
ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/$beamtime

source /etc/profile.d/modules.sh
module purge
source "/home/rodria/scripts/regae/env-regae/bin/activate"

echo "----------------------- Converting pedestals -----------------------"
convert_pedestals.sh ${INP} ${MODE};

echo "----------------------- Creating folders on processed -----------------------"
FOLDER_UP=$(echo $INP | cut -d'/' -f1)
mkdir $ROOT/processed/converted/$FOLDER_UP 2> /dev/null
mkdir $ROOT/processed/assembled/$FOLDER_UP 2> /dev/null

FOLDER=$(echo $INP | cut -d'/' -f2)

mkdir $ROOT/processed/converted/$FOLDER_UP/$FOLDER 2> /dev/null;
mkdir $ROOT/processed/assembled/$FOLDER_UP/$FOLDER 2> /dev/null;

echo "----------------------- Converting images -----------------------"
python ${SCRIPTS_FOLDER}/convert_images.py -p1 ${ROOT}/processed/darks/pedal_d0_${N}_${MODE}_average.h5 -p2 ${ROOT}/processed/darks/pedal_d1_${N}_${MODE}_average.h5 -g1 ${ROOT}/processed/darks/gainMaps_M283.bin -g2 ${ROOT}/processed/darks/gainMaps_M281.bin -i ${ROOT}/raw/${INP} -m 1 -s ${START} -e ${END} -o ${ROOT}/processed/converted/${INP}_master;

python ${SCRIPTS_FOLDER}/save_assembled_images.py -i ${ROOT}/processed/converted/${INP} -g ${ROOT}/scratch_cc/yefanov/geom/JF_regae_v4.geom -o ${ROOT}/processed/assembled/${FOLDER_UP}/${FOLDER};
echo "----------------------- Opening images folder -----------------------"
module load xray;

adxv ${ROOT}/processed/assembled/
