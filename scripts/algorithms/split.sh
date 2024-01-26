#!/bin/sh

INPUT=$1
ROT=$2
mkdir /asap3/petra3/gpfs/p09/2022/data/11016566/processed/rodria/moving_beam_lyso/lyso_0${INPUT}_0${ROT}
split -l50 --numeric-suffixes --suffix-length=2 /asap3/petra3/gpfs/p09/2022/data/11016566/processed/rodria/moving_beam_lyso/lists/lyso_0${INPUT}_0${ROT}.lst /asap3/petra3/gpfs/p09/2022/data/11016566/processed/rodria/moving_beam_lyso/lists/split_lyso_0${INPUT}_0${ROT}.lst

sed -i 's/-rw-r--r-- 1 rodria cfel [0-9][0-9][0-9][0-9][0-9] Nov  [0-9] [0-9][0-9]:[0-9][0-9] //g' h5_files.lst02