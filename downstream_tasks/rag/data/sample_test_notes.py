import pandas as pd 
import numpy as np 
import pickle as pk

import logging
import re, random
import os, argparse
import datasets
from multiprocessing import Pool
from tqdm import tqdm
from itertools import chain

from utils import _clean_deid, _clean_seq

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')



def _sample_mimic_iii(notes_iii, sample_words, SEED):

    # sample pts first
    patients = notes_iii.subject_id.sort_values().unique().tolist()
    random.seed(SEED)
    random.shuffle(patients)

    sampled_patients = []
    n_word = 0

    for patient in patients:
        notes = notes_iii[notes_iii.subject_id == patient]['text'].tolist()
        
        n_word += len(_clean_seq(_clean_deid(' '.join(notes))).split())

        sampled_patients.append(patient)

        if n_word > sample_words:
            break
    
    # sample notes again to check
    ndf = notes_iii[notes_iii.subject_id.isin(sampled_patients)]

    all_notes = ndf.sample(frac=1, random_state=SEED)['text'].tolist()
    
    n_note, n_word = 0, 0
    sampled_notes = []

    for note in all_notes:
        c_note = _clean_seq(_clean_deid(note))
        sampled_notes.append(c_note)
        n_word += len(c_note.split())

        if n_word > sample_words:
            break

    logging.info(f'{n_word} words from {len(sampled_notes)} notes in mimic-iii')

    return sampled_notes, sampled_patients



def _sample_mimic_iv(notes_iv, pt, sample_words, SEED):
    late_groups = ['2014 - 2016', '2017 - 2019', '2020 - 2022']

    ndf = notes_iv.merge(pt, how='left')

    msk1 = ndf.anchor_year_group.isin(late_groups)
    msk2 = pd.to_datetime(ndf.charttime) >= pd.to_datetime(ndf.anchor_year, format='%Y')

    late_notes = ndf[msk1 & msk2]

    ndf = ndf.sample(frac=1, random_state=SEED) # shuffle
    # all_notes = ndf['text'].tolist()

    r_notes = ndf[ndf.category == 'radiology']['text'].tolist()
    d_notes = ndf[ndf.category == 'discharge']['text'].tolist()

    n_note, n_word = 0, 0
    sampled_notes = []

    for note1, note2 in zip(r_notes, d_notes):
        c_note1 = _clean_seq(_clean_deid(note1))
        c_note2 = _clean_seq(_clean_deid(note2))
        
        sampled_notes.append(c_note1)
        sampled_notes.append(c_note2)

        n_word += len(c_note1.split())
        n_word += len(c_note2.split())

        if n_word > sample_words:
            break

        
    logging.info(f'{n_word} words from {len(sampled_notes)} notes in mimic-iv')

    return sampled_notes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--MIMIC_III_DIR", type=str, default='./MIMIC')
    parser.add_argument("--MIMIC_IV_DIR", type=str, default='./MIMIC')

    parser.add_argument("--sample_words", type=int, default=250000)
    parser.add_argument("--debug", "-D", action='store_true', default=False)

    args = parser.parse_args()

    SEED = 42
    TEST_WORDS = args.sample_words

    logging.info(f'loading MIMIC notes')

    if args.debug:
        notes = pd.read_csv(os.path.join(args.MIMIC_III_DIR, 'NOTEEVENTS.csv.gz'), nrows=1000)
        radio = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'radiology.csv.gz'), nrows=1000)
        disch = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'discharge.csv.gz'), nrows=1000)
    else:
        notes = pd.read_csv(os.path.join(args.MIMIC_III_DIR, 'NOTEEVENTS.csv.gz'))
        radio = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'radiology.csv.gz'))
        disch = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'discharge.csv.gz'))

    radio['category'] = 'radiology'
    disch['category'] = 'discharge'

    notes_iv = pd.concat([radio, disch])

    notes.columns = notes.columns.str.lower()

    logging.info(f'MIMIC notes loaded: {len(notes)} iii notes, {len(notes_iv)} iv notes')


    def clean_note_df(df):
        df = df[df['text'].apply(lambda x: len(x.strip())) > 0]
        df = df.drop_duplicates(subset=['text'])
        return df

    # clean
    notes_iii= clean_note_df(notes)
    notes_iv = clean_note_df(notes_iv)

    logging.info(f'After simple cleaning: {len(notes_iii)} iii notes, {len(notes_iv)} iv notes')



    # sample iv: get notes after 2012
    pt_df = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'patients.csv.gz'))

    sampled_notes_iv = _sample_mimic_iv(notes_iv, pt_df, TEST_WORDS, SEED)


    # sample iii: get random pts
    sampled_notes_iii, sampled_patients = _sample_mimic_iii(notes_iii, TEST_WORDS, SEED)


    with open('sampled_pts.txt', 'w') as f:
        for p in sampled_patients:
            f.write(str(p))
            f.write('\n')


    # save datasets.Dataset
    ds_iii = datasets.Dataset.from_dict({'text': sampled_notes_iii})
    ds_iv = datasets.Dataset.from_dict({'text': sampled_notes_iv})

    ds_iii.save_to_disk("test_notes_iii")
    ds_iv.save_to_disk("test_notes_iv")

    logging.info(f'Saved ds')


if __name__ == "__main__":
    main()
