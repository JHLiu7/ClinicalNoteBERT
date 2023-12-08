#!/bin/bash

# conda activate cp

MODEL=jhliu/ClinicalNoteBERT-small-simcse_note

ENCODE_NAME=${MODEL}-enc_file


# encode notes if haven't
# python -u src/encode_notes_bert.py \
#     --model_path $MODEL \
#     --encode_name $ENCODE_NAME 

MODALITY=both

for TASK in mort_hosp los_3 drg; do 
    python -u src/main.py \
        --silent \
        --task $TASK \
        --modality $MODALITY \
        --model_name $MODEL \
        --note_encode_name $ENCODE_NAME
done




