#!/bin/sh

INPUT=$1
START=$2
END=$3
ROOT=/asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria

for i in $(seq $START 1 $END); do
    LABEL=center_${i}
    JNAME="center-p09_${i}"
    NAME="center-p09_${i}"
    SLURMFILE="${NAME}_${INPUT}.sh"
    echo "#!/bin/sh" > $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --partition=allcpu,cfel" >> $SLURMFILE  # Set your partition here
    echo "#SBATCH --time=1-00:00:00" >> $SLURMFILE
    echo "#SBATCH --nodes=1" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --chdir   $PWD" >> $SLURMFILE
    echo "#SBATCH --job-name  $JNAME" >> $SLURMFILE
    echo "#SBATCH --output    $ROOT/error/${NAME}-%N-%j.out" >> $SLURMFILE
    echo "#SBATCH --error     $ROOT/error/${NAME}-%N-%j.err" >> $SLURMFILE
    echo "#SBATCH --nice=0" >> $SLURMFILE
    echo "#SBATCH --mincpus=48" >> $SLURMFILE
    echo "#SBATCH --mem=4G" >> $SLURMFILE
    echo >> $SLURMFILE
    command="./find_center.py -i ${ROOT}/${INPUT}/lists/split_hit_${INPUT}.lst${i} -m  ${ROOT}/${INPUT}/lists/mask_f2x.lst -o ${ROOT}/${INPUT} -g ${ROOT}/geom/${INPUT}/pilatus6M_219mm.geom;"
    echo $command >> $SLURMFILE
    echo "chmod a+rw $PWD" >> $SLURMFILE
    sbatch $SLURMFILE
    mv $SLURMFILE ${ROOT}/shell
done