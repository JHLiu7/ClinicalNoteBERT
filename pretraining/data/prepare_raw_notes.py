import pandas as pd 
import numpy as np 
import pickle as pk

import logging
import re
import os, argparse
from multiprocessing import Pool
from tqdm import tqdm
from itertools import chain

import spacy
from heuristic_tokenize import sent_tokenize_rules
from utils import _clean_deid, _clean_sent

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')

def _fix_category(cat):
    if cat == 'Radiology':
        ncat = 'radiology'
    elif cat == 'Nursing/other' or cat == 'Nursing':
        ncat = 'nursing'
    elif cat == 'Physician ':
        ncat = 'physician'
    elif cat == 'Discharge summary':
        ncat = 'discharge'
    else:
        ncat = 'others'
    return ncat


class SimpleProcessEngine(object):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load('en_core_sci_sm')

    def __call__(self, raw_note):
        """
            raw_note: str
            clean_sent: List, clean_segment: List, clean_note: Str
        """
        clean_sent, clean_segment, clean_note = self.process_note(raw_note)
        return clean_sent, clean_segment, clean_note

    def process_note(self, note):
    
        # note -> seg -> sent
        raw_note = _clean_deid(note)
        raw_segment = sent_tokenize_rules(raw_note)
        raw_sents = [
            self.nlp(seg).sents for seg in raw_segment 
            if set(seg) != {'_'} and set(seg) != {'-'}
        ]
        
        # clean sent
        clean_sents = [
            [_clean_sent(sent.text) for sent in sents] 
            for sents in raw_sents
        ]
        
        # sent -> seg -> note
        clean_sent =  [sent for sent in list(chain(*clean_sents)) if sent.strip() != '']
        clean_segment=[' '.join(sents) for sents in clean_sents]
        clean_note = ' '.join(clean_segment)
        
        return clean_sent, clean_segment, clean_note


def get_sent_seg_note(all_notes, PROC, JOBS):

    # Chunk and put a fix number of words into each process.
    word_len = [len(note.split()) for note in all_notes]
    roll_len = np.cumsum(word_len)
    ALL_WORD_LEN = sum(word_len)
    CHUNKSIZE = int(ALL_WORD_LEN // JOBS)+1

    targets = (roll_len >= (PROC-1) * CHUNKSIZE) & (roll_len < PROC * CHUNKSIZE)
    chunk_notes = [all_notes[i] for i, m in enumerate(targets) if m == True]

    words = sum([word_len[i] for i, m in enumerate(targets) if m == True])

    logging.info(f'Cleaning and processing {PROC}/{JOBS} chunks of {len(chunk_notes)} notes ({words} words) into sent, seg, note')

    engine = SimpleProcessEngine()

    out = []
    for note in tqdm(chunk_notes):
        out.append(engine(note))

    all_sent = [sent for sents in [tup[0] for tup in out] for sent in sents]
    all_segs = [seg for segs in [tup[1] for tup in out] for seg in segs]
    all_note = [tup[2] for tup in out]

    return all_sent, all_segs, all_note


def main():
    """process text into sent, seg, note
    
    The approach: first segment notes, then use spacy to define sentences, then clean sentences.
            Afterwards, resume segments and notes.

    Processing with multiple python commands. Tried multiple ways but either not working as expected or not showing progress.
    Temp fix: chunkize all notes and init multiple python runs to process by bg them. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--MIMIC_III_DIR", type=str, default='./')
    parser.add_argument("--MIMIC_IV_DIR", type=str, default='./')
    parser.add_argument("--DUMP_DIR", type=str, default='TEXT_DUMP')
    
    parser.add_argument("--num_of_job", type=int, default=1, help="[1, total_jobs]")
    parser.add_argument("--total_jobs", type=int, default=8)

    parser.add_argument("--note_type", type=str, default='none', help="['radiology', 'nursing', 'others', 'physician', 'dsummary']")

    parser.add_argument("--debug", "-D", action='store_true', default=False)
    parser.add_argument("--sample", "-S", action='store_true', default=False)

    args = parser.parse_args()

    logging.info(f'loading MIMIC notes')

    if args.debug:
        notes = pd.read_csv(os.path.join(args.MIMIC_III_DIR, 'NOTEEVENTS.csv.gz'), 
                usecols=['SUBJECT_ID', 'CATEGORY', 'TEXT'], nrows=1000)
        radio = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'radiology.csv.gz'), 
                usecols=['subject_id', 'text'], nrows=1000)
        disch = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'discharge.csv.gz'), 
                usecols=['subject_id', 'text'], nrows=1000)
    else:

        notes = pd.read_csv(os.path.join(args.MIMIC_III_DIR, 'NOTEEVENTS.csv.gz'), 
                usecols=['SUBJECT_ID', 'CATEGORY', 'TEXT'])
        radio = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'radiology.csv.gz'), 
                usecols=['subject_id', 'text'])
        disch = pd.read_csv(os.path.join(args.MIMIC_IV_DIR, 'discharge.csv.gz'), 
                usecols=['subject_id', 'text'])

    for d in [radio, disch]:
        d.columns = d.columns.str.upper()
    radio['note_type'] = 'radiology'
    disch['note_type'] = 'discharge'
    notes['note_type'] = notes['CATEGORY'].apply(_fix_category)


    df = pd.concat([notes, radio, disch])

    print(len(notes), len(radio), len(disch))
    print(len(df))

    logging.info(f'MIMIC notes loaded')
    if args.sample: 
        assert args.debug is False
        df.sample(frac=1).head(1000).to_csv('sample_notes.csv', index=False)


    # clean
    df['length'] = df['TEXT'].apply(lambda x: len(x.strip()))
    df = df[df['length']>0]

    # process
    N_TYPE = args.note_type 
    outdir = os.path.join(args.DUMP_DIR, N_TYPE)
    os.makedirs(outdir, exist_ok=True)

    all_notes = df[df['note_type'] == N_TYPE]['TEXT'].drop_duplicates().tolist()
    # all_notes = list(set(all_notes))
    logging.info(f'Process {len(all_notes)} {N_TYPE} notes')

    all_sent, all_segs, all_note = get_sent_seg_note(all_notes, args.num_of_job, args.total_jobs)


    # save
    for all_lines, name in zip([all_sent, all_segs, all_note], ['sentence', 'segment', 'note']):
        num_lines = len(all_lines)
        num_tokens = len(' '.join(all_lines).split())
        logging.info(f'Obtained {num_lines:,} lines of {name}: in total {num_tokens:,} tokens')

        line_dir = os.path.join(outdir, name)
        os.makedirs(line_dir, exist_ok=True)
        outpath = os.path.join(line_dir, f'{args.num_of_job:0>2d}_{args.total_jobs:0>2d}.txt')

        with open(outpath, 'w') as f:
            for line in all_lines:
                f.write(line)
                f.write('\n')

        logging.info(f'Dumped text to {line_dir}')


if __name__ == "__main__":
    main()



