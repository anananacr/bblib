#!/bin/sh

PATH=/asap3/petra3/gpfs/p09/2023/data/11016750/processed/rodria/f2x_D_01_01/cluster_2
N=500

/usr/bin/ls $PATH| /usr/bin/sort -R| /usr/bin/tail -$N | while read file; do
/usr/bin/zip $PATH/../cluster_2.zip $PATH/$file 
done
