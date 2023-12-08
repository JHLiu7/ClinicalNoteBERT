#!/bin/bash



MODEL=jhliu/ClinicalNoteBERT-base-note_ntp



ROOT_DIR=./

cd $ROOT_DIR

DATA_DIR=$ROOT_DIR/data/
OUTPUT=$ROOT_DIR/results/
CACHE=$ROOT_DIR/cache/

SCHEDULE=constant
WARMUP=0.1


# NLI
DATA_NAME=MedNLI
for LR in 2e-5 3e-5 5e-5; do 
    for EPOCH in 3 4 5; do
        for BSIZE in 16 32; do

            python src/run_nli.py --model_name_or_path $MODEL \
                --dataset_name $DATA_NAME \
                --dataset_dir $DATA_DIR \
                --output_dir $OUTPUT \
                --cache_dir $CACHE \
                --do_train --do_eval --do_predict \
                --per_device_train_batch_size $BSIZE \
                --per_device_eval_batch_size $BSIZE \
                --save_strategy no \
                --num_train_epochs $EPOCH \
                --learning_rate $LR \
                --lr_scheduler_type $SCHEDULE \
                --warmup_ratio $WARMUP \
                --max_seq_length 128

        done
    done
done


# NER 
for DATA_NAME in i2b2_2010 i2b2_2012; do

    for LR in 2e-5 3e-5 5e-5; do 
        for EPOCH in 3 4 5; do
            for BSIZE in 16 32; do

                python src/run_ner.py --model_name_or_path $MODEL \
                    --dataset_name $DATA_NAME \
                    --dataset_dir $DATA_DIR \
                    --output_dir $OUTPUT \
                    --cache_dir $CACHE \
                    --do_train --do_eval --do_predict \
                    --per_device_train_batch_size $BSIZE \
                    --per_device_eval_batch_size $BSIZE \
                    --save_strategy no \
                    --num_train_epochs $EPOCH \
                    --learning_rate $LR \
                    --lr_scheduler_type $SCHEDULE \
                    --warmup_ratio $WARMUP \
                    --max_seq_length 150

            done
        done
    done
done


# QA
DATA_NAME=radqa
for LR in 2e-5 3e-5 5e-5; do 
    for EPOCH in 3 4 5; do
        for BSIZE in 16 32; do
            python src/run_qa.py --model_name_or_path $MODEL \
                --dataset_name $DATA_NAME \
                --dataset_dir $DATA_DIR \
                --output_dir $OUTPUT \
                --cache_dir $CACHE \
                --do_train --do_eval --do_predict \
                --per_device_train_batch_size $BSIZE \
                --per_device_eval_batch_size $BSIZE \
                --save_strategy no \
                --num_train_epochs $EPOCH \
                --learning_rate $LR \
                --lr_scheduler_type $SCHEDULE \
                --warmup_ratio $WARMUP \
                --max_seq_length 384 \
                --doc_stride 128
        done
    done
done

DATA_NAME=emrqa_medication
for LR in 3e-5 5e-5; do 
    for EPOCH in 3 5; do
        for BSIZE in 32; do
            python src/run_qa.py --model_name_or_path $MODEL \
                --dataset_name $DATA_NAME \
                --dataset_dir $DATA_DIR \
                --output_dir $OUTPUT \
                --cache_dir $CACHE \
                --do_train --do_eval --do_predict \
                --per_device_train_batch_size $BSIZE \
                --per_device_eval_batch_size $BSIZE \
                --save_strategy no \
                --num_train_epochs $EPOCH \
                --learning_rate $LR \
                --lr_scheduler_type $SCHEDULE \
                --warmup_ratio $WARMUP \
                --max_seq_length 384 \
                --doc_stride 128
        done
    done
done


