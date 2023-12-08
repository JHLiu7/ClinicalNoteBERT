#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

#SBATCH --time=08:00:00


NTYPE=radiology


## tokenize
MODEL_PATH=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
JOBS=$SLURM_CPUS_PER_TASK


PTYPE=sentence
MIN_LEN=16
MAX_LEN=128


python -u tokenize_lines.py \
    --tokenizer_path $MODEL_PATH \
    --note_category $NTYPE \
    --process_type $PTYPE \
    --min_length $MIN_LEN \
    --max_length $MAX_LEN \
    --n_jobs $JOBS 
    

PTYPE=segment
MIN_LEN=128
MAX_LEN=256


python -u tokenize_lines.py \
    --tokenizer_path $MODEL_PATH \
    --note_category $NTYPE \
    --process_type $PTYPE \
    --min_length $MIN_LEN \
    --max_length $MAX_LEN \
    --n_jobs $JOBS 
    

PTYPE=note
MIN_LEN=256
MAX_LEN=512


python -u tokenize_lines.py \
    --tokenizer_path $MODEL_PATH \
    --note_category $NTYPE \
    --process_type $PTYPE \
    --min_length $MIN_LEN \
    --max_length $MAX_LEN \
    --n_jobs $JOBS 
    
