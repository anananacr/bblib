#!/bin/sh
#
# Convert pedestal images using om_jungfrau_dark.py script from Onda monitor.
# Remember to set correctly paths where pedestals from JUNGFRAU 1M are stored.
#
# ./convert_pedestal.sh input
#./convert_pedestal.sh 231020_mica_c4_m1_001/ed_rot_step_001/231020_mica_c4_m1_001_001 step

# Written by Ana Carolina Rodrigues.
#
# Mail: ana.rodrigues@desy.de

source /home/rodria/scripts/regae/env-regae/bin/activate

INP=$1
MODE=$2
N=$(basename $INP)
beamtime=11018148
OUTPUT=/asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/processed/darks
echo $N

## convert start dark
ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/${INP}*_start_pedal*_*d0_f*.h5>pedal_d0.lst
/home/rodria/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d0.lst ${OUTPUT}/pedal_d0_${N}_${MODE}_start.h5;

ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/${INP}*_start_pedal*_*d1_f*.h5>pedal_d1.lst
/home/rodria/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d1.lst ${OUTPUT}/pedal_d1_${N}_${MODE}_start.h5;

## average dark
if [ "$MODE" == "step" ];
then
## convert stop dark
  ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/${INP}*_stop_pedal*_*d0_f*.h5>pedal_d0.lst
  /home/rodria/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d0.lst ${OUTPUT}/pedal_d0_${N}_${MODE}_stop.h5;

  ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/${INP}*_stop_pedal*_*d1_f*.h5>pedal_d1.lst
  /home/rodria/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d1.lst ${OUTPUT}/pedal_d1_${N}_${MODE}_stop.h5;
  python merge_pedal.py -l ${N}_${MODE}
elif [ "$MODE" == "magnet" ];
then
  cp ${OUTPUT}/pedal_d0_${N}_${MODE}_start.h5 ${OUTPUT}/pedal_d0_${N}_${MODE}_average.h5
  cp ${OUTPUT}/pedal_d1_${N}_${MODE}_start.h5 ${OUTPUT}/pedal_d1_${N}_${MODE}_average.h5
elif [ "$MODE" == "screening" ];
then
  cp ${OUTPUT}/pedal_d0_${N}_${MODE}_start.h5 ${OUTPUT}/pedal_d0_${N}_${MODE}_average.h5
  cp ${OUTPUT}/pedal_d1_${N}_${MODE}_start.h5 ${OUTPUT}/pedal_d1_${N}_${MODE}_average.h5
else
  ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/${INP}*_stop_pedal*_*d0_f*.h5>pedal_d0.lst
  /home/rodria/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d0.lst ${OUTPUT}/pedal_d0_${N}_${MODE}_stop.h5;
  
  ls /asap3/fs-bmx/gpfs/regae/2023/data/$beamtime/raw/${INP}*_stop_pedal*_*d1_f*.h5>pedal_d1.lst
  /home/rodria/software/om_dev_regae/om/bin_src/om_jungfrau_dark.py pedal_d1.lst ${OUTPUT}/pedal_d1_${N}_${MODE}_stop.h5;
  python merge_pedal.py -l ${N}_${MODE}
fi

