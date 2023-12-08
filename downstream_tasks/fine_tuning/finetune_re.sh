#!/bin/bash


# conda activate re

MODEL=jhliu/ClinicalNoteBERT-base-note_ntp


DATA_DIR=$ROOT_DIR/data/ # radgraph folder
OUTPUT=$ROOT_DIR/results/

python src/FT/run_re.py --model_name_or_path $MODEL \
    --dataset_dir $DATA_DIR \
    --output_dir $OUTPUT 
    
