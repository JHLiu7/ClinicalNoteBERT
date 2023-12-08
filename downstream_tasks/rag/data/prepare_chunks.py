import pandas as pd 
import numpy as np 
import pickle as pk

import logging
import re, random
import os, argparse
from multiprocessing import Pool
from tqdm import tqdm
from itertools import chain
import datasets

from heuristic_tokenize import sent_tokenize_rules
from utils import _clean_deid, _clean_seq

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')


def process_note(note):

    # note -> seg -> sent
    raw_note = _clean_deid(note)
    clean_segment = [_clean_seq(seg) for seg in 
        sent_tokenize_rules(raw_note)
        if set(seg) != {'_'} and set(seg) != {'-'}
    ]
    
    return clean_segment


def chunk_sequences(all_seqs, MIN_LEN, MAX_LEN):
    chunks = []
    for seq in tqdm(all_seqs):
        words = seq.split()
        nlen = len(words)
        if nlen > MIN_LEN:
            s = words[:MAX_LEN]
            chunks.append(' '.join(s))

    chunks = sorted(list(set(chunks)), key=len, reverse=True)

    return chunks




def chunk_segments_v2(all_seqs):
    MIN_SEG_LEN = 128
    MAX_SEG_LEN = 256

    segs = []
    run_s = ''

    for seq in tqdm(all_seqs):
        nlen = len(seq.split())
        if nlen < MIN_SEG_LEN:
            if nlen >= MIN_SENT_LEN:
                run_s = ' '.join([run_s, seq])
            if len(run_s) > MAX_SEG_LEN:
                segs.append(run_s)
                run_s = ''
        else:
            if nlen <= MAX_SEG_LEN:
                segs.append(seq)
            else:
                num_seq = nlen // MAX_SEG_LEN
                words = seq.split()
                for i in range(0, num_seq):
                    segs.append(
                        ' '.join(words[i*MAX_SEG_LEN : (i+1)*MAX_SEG_LEN])
                    )
    return segs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--MIMIC_III_DIR", type=str, default='./MIMIC')

    parser.add_argument("--sampled_patients", type=str, default='sampled_pts.txt')
    parser.add_argument("--debug", "-D", action='store_true', default=False)

    args = parser.parse_args()

    # prepare chunks for datastore

    df = pd.read_csv(os.path.join(args.MIMIC_III_DIR, 'NOTEEVENTS.csv.gz'))

    df = df[df['TEXT'].apply(lambda x: len(x.strip())) > 0]
    df = df.drop_duplicates(subset=['TEXT'])

    if args.debug: df = df.head(1000)

    logging.info(f"Loaded raw mimic notes: {len(df)}")


    # remove test pts
    sampled_patients = [
        int(p.strip()) for p in open(args.sampled_patients, 'r').readlines()
    ]
    all_notes = df[~df.SUBJECT_ID.isin(sampled_patients)]['TEXT'].tolist()

    logging.info(f"After removing notes of test pts: {len(all_notes)}")


    # process seqs
    all_seqs = []

    for note in tqdm(all_notes):
        all_seqs.extend(process_note(note))

    logging.info(f"Obtained {len(all_seqs)} lines")
    

    # get sents

    # sents = chunk_sentences(all_seqs)
    sents = chunk_sequences(all_seqs, 16, 128)
    logging.info(f"Obtained {len(sents)} sents")


    # get segs

    # segs = chunk_segments(all_seqs)
    segs = chunk_sequences(all_seqs, 128, 256)
    logging.info(f"Obtained {len(segs)} segments")


    # save datasets.Dataset
    ds_sent = datasets.Dataset.from_dict({'text': sents})
    ds_seg  = datasets.Dataset.from_dict({'text': segs })

    ds_sent.save_to_disk("chunks_sent")
    ds_seg.save_to_disk( "chunks_seg" )

    logging.info(f'Saved ds')


if __name__ == "__main__":
    main()
