#!/bin/bash



python sample_test_notes.py \
    --MIMIC_III_DIR $MIMICIII \
    --MIMIC_IV_DIR $MIMICIV

python prepare_chunks.py \
    --MIMIC_III_DIR $MIMICIII \
    --sampled_patients sampled_pts.py 

