#!/bin/sh

#SBATCH --partition=allcpu,cfel
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem=8G

#SBATCH --job-name  convert_jf
#SBATCH --output /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/stability_measurements/error/231130_mica-%N-%j.out
#SBATCH --error /asap3/fs-bmx/gpfs/regae/2023/data/11018148/scratch_cc/rodria/stability_measurements/error/231130_mica-%N-%j.err


# Convert images using convert_all.py script. Remember to set correctly paths where data and pedestals from JUNGFRAU 1M are stored.
# Enable H5 data to be converted in step-wise manner, using start and end frame number arguments,  dividing it in smaller jobs.

# ./convert_step.sh 231020_mos_c3_ms_001/ed_rot_step_003/231020_mos_c3_ms_001_003 step 0 12999

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /etc/profile.d/modules.sh
module load maxwell python/3.7
source /home/rodria/scripts/regae/env-regae/bin/activate

#INPUT=231020_mos_c3_ms_001/ed_rot_step_003/231020_mos_c3_ms_001_003
#PEDAL=231020_mos_c3_ms_003_average

INP=$1
MODE=$2
START=$3
END=$4

N=$(basename $INP)

if [ "$MODE" == "step" ];
then
  MODE_NUMBER=1
elif [ "$MODE" == "magnet" ];
then
  MODE_NUMBER=1
elif [ "$MODE" == "screening" ];
then
  MODE_NUMBER=1
else
  MODE_NUMBER=0
fi

echo $MODE_NUMBER
ROOT=/asap3/fs-bmx/gpfs/regae/2023/data/11018148/processed/converted

python3 convert_images.py -p1 ${ROOT}/../darks/pedal_d0_${N}_${MODE}_average.h5 -p2 ${ROOT}/../darks/pedal_d1_${N}_${MODE}_average.h5 -g1 ${ROOT}/../darks/gainMaps_M283.bin -g2 ${ROOT}/../darks/gainMaps_M281.bin -i ${ROOT}/../../raw/${INP} -m ${MODE_NUMBER} -s ${START} -e ${END} -o ${ROOT}/${INP}_master;