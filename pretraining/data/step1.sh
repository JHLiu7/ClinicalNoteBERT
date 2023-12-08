#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --output=slurm-out/slurm-%x-%A.out

#SBATCH --time=08:00:00
#SBATCH --array=1-12


NTYPE=radiology

J=$SLURM_ARRAY_TASK_ID
T=$SLURM_ARRAY_TASK_COUNT


DIR_III=/path/to/mimiciii
DIR_IV=/path/to/mimiciv

OUT_DIR=RAW_TEXT

python -u prepare_raw_notes.py \
    --MIMIC_III_DIR $DIR_III \
    --MIMIC_IV_DIR $DIR_IV \
    --DUMP_DIR $OUT_DIR \
    --num_of_job $J \
    --total_jobs $T \
    --note_type $NTYPE



    
