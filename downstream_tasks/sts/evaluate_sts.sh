#!/bin/bash

MODEL=jhliu/ClinicalNoteBERT-base-simcse_sentence

python src/eval_sts.py \
    --sts_data_path $STS_DATA \
    --model_path $MODEL


# Train SimCSE models
# MODEL=jhliu/ClinicalNoteBERT-base-note_only
# python src/run_pipeline.py $MODEL

