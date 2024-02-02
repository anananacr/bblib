#!/bin/sh
FOLDER=$1
SAMPLE=$2
ROOT=/asap3/petra3/gpfs/p09/2023/data/11019088/processed/rodria
mkdir ${ROOT}/streams/${FOLDER}
if [ "$SAMPLE" -eq 0 ];
then
    sbatch batch_index_lyso.sh lyso_centered_all_events ${FOLDER}/lyso_centered_all_events_center_no_cell 1 1 0
    sbatch batch_index_lyso.sh lyso_centered_all_events ${FOLDER}/lyso_centered_all_events_no_center_no_cell 0 1 0
    sbatch batch_index_lyso.sh lyso_centered_all_events ${FOLDER}/lyso_centered_all_events_center_latt 1 1 0 lyso_latt.cell
    sbatch batch_index_lyso.sh lyso_centered_all_events ${FOLDER}/lyso_centered_all_events_center_latt_cell 1 1 0 lyso.cell
    sbatch batch_index_lyso.sh lyso_centered_all_events ${FOLDER}/lyso_centered_all_events_no_center_latt_cell 0 1 0 lyso.cell
    sbatch batch_index_lyso.sh lyso_centered_all_events ${FOLDER}/lyso_centered_all_events_no_center_latt 0 1 0 lyso_latt.cell
else
    sbatch batch_index_fosakp.sh fakp_centered_all_events ${FOLDER}/fakp_centered_all_events_center_no_cell 1 1 0
    sbatch batch_index_fosakp.sh fakp_centered_all_events ${FOLDER}/fakp_centered_all_events_no_center_no_cell 0 1 0
    sbatch batch_index_fosakp.sh fakp_centered_all_events ${FOLDER}/fakp_centered_all_events_center_latt 1 1 0 fakp_latt.cell
    sbatch batch_index_fosakp.sh fakp_centered_all_events ${FOLDER}/fakp_centered_all_events_center_latt_cell 1 1 0 fakp.cell
    sbatch batch_index_fosakp.sh fakp_centered_all_events ${FOLDER}/fakp_centered_all_events_no_center_latt_cell 0 1 0 fakp.cell
    sbatch batch_index_fosakp.sh fakp_centered_all_events ${FOLDER}/fakp_centered_all_events_no_center_latt 0 1 0 fakp_latt.cell
fi
