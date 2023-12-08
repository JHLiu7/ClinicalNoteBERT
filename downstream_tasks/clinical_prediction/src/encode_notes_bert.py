import logging
import os

import torch
import datasets
import numpy as np
import pandas as pd
import pickle as pk

from tqdm import tqdm

import argparse 

import transformers
import datasets

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


def encode_df(tokenizer, model, note_df, pool_type='cls', batch_size=128, USE_CUDA=True):

    notes = note_df['CLEAN_TEXT'].tolist()
    input_key = "text"
    raw_dataset = datasets.Dataset.from_dict({input_key: notes})

    def tokenize_function(examples):
        result = tokenizer(examples[input_key], padding=True, max_length=512, truncation=True)
        return result

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=[input_key]
    )
    tokenized_dataset.set_format(type="torch")

    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size)
    num_batch = int(len(tokenized_dataset) / batch_size)

    cls_list = []

    for batch in tqdm(data_loader, total=num_batch):
        if USE_CUDA:
            batch = {k:v.cuda() for k,v in batch.items()}
        
        with torch.no_grad():
            out = model(**batch)

        if pool_type == 'cls':
            cls = out.last_hidden_state[:, 0].cpu()
        elif pool_type == 'avg':
            cls = out.last_hidden_state.mean(1).cpu()

        cls_list.append(cls)

    all_cls = torch.cat(cls_list, 0)
    dim = all_cls.size(-1)

    # assign to df
    cls_ready = [i for i in all_cls.numpy()]
    assert len(notes) == len(cls_ready)

    note_df['vector'] = cls_ready

    note_df = note_df[note_df.columns.drop(['CLEAN_TEXT'])]

    return note_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--note_df_folder", default='/data/scratch/projects/punim1362/jinghuil1/TmpCBERT/data/EB/clinical_pred/data_notes', type=str, help='input dir')

    parser.add_argument("--model_path", default='', type=str)
    parser.add_argument("--encode_name", default='', type=str)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--pool_type", default='cls', type=str)

    parser.add_argument("--output_dir", default='/data/scratch/projects/punim1362/jinghuil1/TmpCBERT/data/EB/clinical_pred/data_notes_encoded', type=str, help='output dir')
    parser.add_argument("--bert_cache_dir", default='/data/scratch/projects/punim1362/jinghuil1/TmpCBERT/cache/encode', type=str)

    parser.add_argument('--debug', '-D', action="store_const", const=True, default=False)
    args = parser.parse_args()

    USE_CUDA = True

    print(os.listdir(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.bert_cache_dir)
    model = AutoModel.from_pretrained(args.model_path, cache_dir=args.bert_cache_dir)
    model.eval()
    if USE_CUDA:
        model.cuda()

    for cohort in ['mort_los', 'drg']:
        inpath = os.path.join(args.note_df_folder, f'{cohort}_df.p')
        note_df = pd.read_pickle(inpath)
        if args.debug:
            note_df = note_df.head(1000)
        logging.info(f"Loaded {len(note_df)} notes from {inpath}")

        encoded_df = encode_df(tokenizer, model, note_df, args.pool_type, args.batch_size)

        # dump 
        fname = f"{args.encode_name}-{cohort}.p"
        os.makedirs(args.output_dir, exist_ok=True)
        outpath = os.path.join(args.output_dir, fname)
        with open(outpath, 'wb') as outf:
            pk.dump(encoded_df, outf)

        logging.info(f"Dumped data to {outpath}")

if __name__ == '__main__':
    main()


