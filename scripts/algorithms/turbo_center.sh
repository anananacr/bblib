#!/bin/sh

INPUT=$1
START=$2
END=$3

## SET HERE YOUR PATHS
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria
SCRATCH=/asap3/petra3/gpfs/p09/2023/data/11019088/scratch_cc/rodria


for i in $(seq $START 1 $END); do
    if [ "$i" -le 9 ];
    then
        LIST_NAME=${INPUT}.lst0${i}
    else 
        LIST_NAME=${INPUT}.lst${i}
    fi
    LABEL=center_${i}
    JNAME="center_${i}"
    NAME="center_${i}"
    SLURMFILE="${NAME}_${INPUT}.sh"
    echo "#!/bin/sh" > $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --partition=allcpu,cfel" >> $SLURMFILE  # Set your partition here
    echo "#SBATCH --time=2-01:00:00" >> $SLURMFILE
    echo "#SBATCH --nodes=1" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --chdir   $PWD" >> $SLURMFILE
    echo "#SBATCH --job-name  $JNAME" >> $SLURMFILE
    echo "#SBATCH --requeue" >> $SLURMFILE
    echo "#SBATCH --output    /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/${NAME}-%N-%j.out" >> $SLURMFILE
    echo "#SBATCH --error     /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/error/${NAME}-%N-%j.err" >> $SLURMFILE
    echo "#SBATCH --nice=0" >> $SLURMFILE
    echo "#SBATCH --mincpus=128" >> $SLURMFILE
    echo "#SBATCH --mem=40G" >> $SLURMFILE
    echo "#SBATCH --mail-type=ALL" >> $SLURMFILE
    echo "#SBATCH --mail-user=errodriguesana@gmail.com" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "unset LD_PRELOAD" >> $SLURMFILE
    echo "source /etc/profile.d/modules.sh" >> $SLURMFILE
    echo "module purge" >> $SLURMFILE
    echo "source /home/rodria/scripts/p09/env-p09/bin/activate" >> $SLURMFILE
    echo >> $SLURMFILE
    command="python find_center.py -i ${ROOT}/lists/${LIST_NAME} -o ${ROOT} -s ${SCRATCH} -m /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/mask/mask_ana_dec_fakp.h5 -g /asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria/geoms/eiger500k_corrected_beam_centre_fosakp.geom;"
    echo $command >> $SLURMFILE
    echo "chmod a+rw $PWD" >> $SLURMFILE
    sbatch $SLURMFILE 
    mv $SLURMFILE ${ROOT}/shell
done