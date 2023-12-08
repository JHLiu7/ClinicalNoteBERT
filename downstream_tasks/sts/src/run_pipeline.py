import subprocess, os, argparse, sys
import numpy as np

from eval_sts import evaluate_sts

os.environ["TOKENIZERS_PARALLELISM"]="false"

def main(MODEL):

    ROOT_DIR = '/data/scratch/projects/punim1362/jinghuil1/TmpCBERT/'
    model_name = _get_abbrv_from_model_name_or_path(MODEL)

    # raw
    test_score = evaluate_sts(MODEL, 
            os.path.join(ROOT_DIR, 'data/FT/ClinicalSTS'), 
            'dev')

    output_res(ROOT_DIR, model_name, 'raw', MODEL, test_score, None, None, None, None)

    # seqs
    for SEQ_TYPE in ['sentence', 'segment', 'note']:
        (best_config, best_val_score), (configs, val_scores) = tune_simcse(MODEL, SEQ_TYPE)
        STEPS, LR = best_config

        # re train model
        CKPT = train_simcse(
            MODEL=MODEL,
            SEQ_TYPE=SEQ_TYPE,
            STEPS=STEPS,
            LR=LR,
            PHASE='best'
        )

        # eval and output
        test_score = evaluate_sts(CKPT, 
            os.path.join(ROOT_DIR, 'data/FT/ClinicalSTS'), 
            'dev')

        output_res(ROOT_DIR, model_name, SEQ_TYPE, CKPT, test_score, configs, val_scores, best_config, best_val_score)



def tune_simcse(MODEL, SEQ_TYPE):

    configs, scores = [], []
    for STEPS in [100, 500, 1000]:
    # for STEPS in [10]:
        for LR in [2e-5, 3e-5, 5e-5]:
        # for LR in [2e-5]:

            CKPT = train_simcse(MODEL, SEQ_TYPE, STEPS, LR)

            score = evaluate_sts(CKPT, 
                '/data/scratch/projects/punim1362/jinghuil1/TmpCBERT/data/FT/ClinicalSTS', 
                'train_sample')

            configs.append([STEPS, LR])
            scores.append(score)

    ix = np.argmax(scores)
    best_config = configs[ix]
    best_score = scores[ix]

    return (best_config, best_score), (configs, scores)



def _get_abbrv_from_model_name_or_path(model_path):

    if os.path.isdir(model_path):
        model_name = model_path.strip('/').split('/')[-1]

    if model_name == 'bert-base-cased':
        name = 'bert'
    elif model_name == '':
        name = 'ClinicalBERT'
    elif model_name == 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':
        name = 'PubMedBERT'
    else:
        name = model_name

    return name 

def train_simcse(MODEL, SEQ_TYPE, STEPS, LR=3e-5, OUTPUT_DIR='SimCSE', PHASE='tune'):
    
    if SEQ_TYPE == 'sentence':
        BATCH_SIZE = 128
        GRAD_ACC = 1
    elif SEQ_TYPE == 'segment':
        BATCH_SIZE = 32
        GRAD_ACC = 4
    elif SEQ_TYPE == 'note':
        BATCH_SIZE = 16
        GRAD_ACC = 8

    name = _get_abbrv_from_model_name_or_path(MODEL)
    CKPT = os.path.join(OUTPUT_DIR, name, f"{SEQ_TYPE}-{PHASE}")

    if 'large' in name.lower():
        BATCH_SIZE = int(BATCH_SIZE / 4)
        GRAD_ACC *= 2

    CMD_simcse = f"python src/train_simcse.py \
        --model_name_or_path {MODEL} \
        --output_dir {CKPT} \
        --sequence_type {SEQ_TYPE} \
        --max_steps {STEPS} \
        --per_device_train_batch_size {BATCH_SIZE} \
        --per_device_eval_batch_size {BATCH_SIZE} \
        --gradient_accumulation_steps {GRAD_ACC} \
        --cache_dir cache \
        --raw_text_dir data/PT_v2/RAW_TEXT \
        --do_train True \
        --save_steps {STEPS} \
        --learning_rate {LR} \
        --weight_decay 0.01 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --overwrite_output_dir True"

    process = subprocess.Popen(CMD_simcse.split(), stdout=subprocess.PIPE, cwd=".")
    output, error = process.communicate()

    return CKPT



def output_res(ROOT_DIR, model_name, SEQ_TYPE, CKPT, test_score, configs, val_scores, best_config, best_val_score):

    line0 = f"{model_name} at {CKPT}\n\n"

    line1 = f"configs {str(configs)} and scores {str(val_scores)}\n"
    line2 = f"best config {str(best_config)} and best val score {best_val_score}\n\n"

    line3 = f"Test score: {test_score}\n"

    folder = os.path.join(ROOT_DIR, 'results', model_name, 'STS')
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, f'res-{SEQ_TYPE}.txt'), 'w') as f:
        f.write(line0)
        f.write(line1)
        f.write(line2)
        f.write(line3)





if __name__ == "__main__":
    main(sys.argv[1])


