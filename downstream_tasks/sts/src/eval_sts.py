import logging
import os

import torch
import datasets
import numpy as np

import argparse
import random
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_raw_file(path, split):
    def _one_file(f):
        lines = open(f, 'r').readlines()
        t1, t2, s = [], [], []
        for l in lines:
            a,b,c = l.strip().split("\t")
            t1.append(a)
            t2.append(b)
            s.append(c)
        return t1, t2, s

    train_lines = _one_file(os.path.join(path, 'train.txt'))
    dev_lines = _one_file(os.path.join(path, 'dev.txt'))

    if split == 'train':
        return train_lines
    elif split == 'dev':
        return dev_lines
    elif split == 'both':
        tr_l1, tr_l2, tr_s = train_lines
        dev_l1, dev_l2, dev_s = dev_lines
        b_l1 = tr_l1 + dev_l1
        b_l2 = tr_l2 + dev_l2
        b_s = tr_s + dev_s
        return (b_l1, b_l2, b_s)
    elif split == 'train_sample':
        t1, t2, s = train_lines
        NUM=200
        out = []
        for l in [t1, t2, s]:
            random.seed(42)
            out.append(random.sample(l, NUM))
        return out


def evaluate_sts(model_path, sts_data_path, split, batch_size=64):

    USE_CUDA = True

    # read raw text 
    lines_set1, lines_set2, scores = read_raw_file(sts_data_path, split)

    # load model
    if os.path.isdir(model_path):
        cache_dir=None
    else:
        cache_dir='tmp'
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir) #, torch_dtype=torch.float16, device_map='auto') #.eval()
    model.eval()
    if USE_CUDA:
        model.cuda()

    if 't5' in model_path.lower():
        MEAN_POOLING=True
        model = model.encoder
    else:
        MEAN_POOLING=False

    # encode line set 
    def encode_line(lines):
        input_key = 'text'
        raw_dataset = datasets.Dataset.from_dict({input_key: lines})

        def tokenize_function(examples):
            result = tokenizer(examples[input_key], padding="max_length", max_length=128, truncation=True)
            return result

        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True,
            desc="Running tokenizer on dataset", remove_columns=[input_key])
        tokenized_dataset.set_format(type="torch")

        data_loader = DataLoader(tokenized_dataset, batch_size=batch_size)
        num_batch = int(len(tokenized_dataset) / batch_size)

        rep_list = []
        for batch in tqdm(data_loader, total=num_batch):
            if USE_CUDA:
                batch = {k:v.cuda() for k,v in batch.items()}
            
            with torch.no_grad():
                out = model(**batch)

            if MEAN_POOLING:
                rep = out.last_hidden_state.mean(1).cpu()
            else:
                rep = out.last_hidden_state[:, 0].cpu()

            rep_list.append(rep)

        all_rep = torch.cat(rep_list, 0)
        return all_rep


    embed_set1 = encode_line(lines_set1)
    embed_set2 = encode_line(lines_set2)
        
    # calculate scores 
    cos_sim = torch.nn.CosineSimilarity()
    pred_scores = cos_sim(embed_set1, embed_set2)

    # eval
    true_scores = np.array([float(i) for i in scores])
    pred = pred_scores.numpy()
    res = spearmanr(pred, true_scores)
    
    pr = pearsonr(pred, true_scores)[0]

    return pr
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sts_data_path", default='/data/scratch/projects/punim1362/jinghuil1/TmpCBERT/data/FT/ClinicalSTS', type=str)
    
    parser.add_argument("--split", default='dev', type=str)

    parser.add_argument("--model_path", "-M", default='', type=str)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--batch_size", default=64, type=int)

    args = parser.parse_args()

    
    pr = evaluate_sts(args.model_path, args.sts_data_path, args.split)
    

    print()

    print(args.model_name)
    print(f'Correlation on {args.split}')
    print(f"{pr*100:.2f}")

    print()

if __name__ == '__main__':
    main()


