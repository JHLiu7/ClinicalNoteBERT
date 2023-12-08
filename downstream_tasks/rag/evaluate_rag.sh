#!/bin/bash


# conda activate rag


RT_MODEL=jhliu/ClinicalNoteBERT-base-note_only-simcse_segment




DS_SEG_PATH=data/chunks_seg
DS_DATA_PATH=$DS_SEG_PATH

echo $RT_MODEL
echo $DS_SEG_PATH


python src/create_embeddings.py --retriever_model_name_or_path $RT_MODEL --datastore_path $DS_SEG_PATH 
python src/create_index.py --retriever_model_name_or_path $RT_MODEL --datastore_path $DS_SEG_PATH 


for EVAL_SET in iii iv
do

    EVAL_DATA_PATH=data/test_notes_${EVAL_SET}

    for LM_MODEL in meta-llama/Llama-2-13b-hf meta-llama/Llama-2-7b-hf
    do
        python src/eval_ppl.py \
            --lm_model_name_or_path $LM_MODEL \
            --retriever_model_name_or_path $RT_MODEL \
            --datastore_path $DS_DATA_PATH \
            --eval_data_path $EVAL_DATA_PATH \
            --write_results \
            --stride 512 --max_length 1024
    done

    for LM_MODEL in gpt2-xl gpt2
    do
        python src/eval_ppl.py \
            --lm_model_name_or_path $LM_MODEL \
            --retriever_model_name_or_path $RT_MODEL \
            --datastore_path $DS_DATA_PATH \
            --eval_data_path $EVAL_DATA_PATH \
            --write_results \
            --stride 256 --max_length 512
    done

done

