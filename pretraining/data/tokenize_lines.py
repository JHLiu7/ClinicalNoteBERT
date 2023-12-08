import pandas as pd 
import numpy as np 
import pickle as pk

import logging
import os, argparse, json
from itertools import chain

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')


class TokenizeEngine():
    def __init__(self, tokenizer, text_column_name='text', add_special_tokens=False) -> None:
        self.tokenizer = tokenizer
        self.text_column_name = text_column_name
        self.add_special_tokens = add_special_tokens

    def __call__(self, examples):
        
        examples[self.text_column_name] = [
            line for line in examples[self.text_column_name] if len(line) > 0 and not line.isspace()
        ]
        
        return self._tokenize(examples=examples[self.text_column_name])

    def _tokenize(self, examples):
        
        if self.add_special_tokens:
            return_dict = self.tokenizer(examples, return_special_tokens_mask=True)
        else:
            # will add them later manually to make sure [cls] and [sep] are at ends
            return_dict = self.tokenizer(examples, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, return_special_tokens_mask=False)
        
        return_dict.update({'length': [len(i) for i in return_dict['input_ids']]})
        
        return return_dict


class LabelEngine:
    def __init__(self, note_category):

        NOTE_CATEGORIES = ['discharge', 'radiology', 'nursing', 'physician', 'others']
        NOTE2ID = {t:i for i,t in enumerate(NOTE_CATEGORIES)}

        assert note_category in NOTE_CATEGORIES

        self.label = NOTE2ID[note_category] 

    def __call__(self, examples):
        examples['note_category'] = self.label
        return examples


class ProcessEngine():
    def __init__(self, process_type, MIN_SEQ_LEN, MAX_SEQ_LEN, CLS_ID, SEP_ID):
        # process_type: sentence, segment, note
        # note adds category label
        
        self.process_type = process_type
        self.MIN_SEQ_LEN = MIN_SEQ_LEN
        self.MAX_SEQ_LEN = MAX_SEQ_LEN

        # add [cls] and [sep]
        assert CLS_ID is not None and SEP_ID is not None
        
        self.MAX_SEQ_LEN -= 2
        self.CLS_ID = CLS_ID
        self.SEP_ID = SEP_ID

        self.MIN_SENT_LEN=16 # throw away sent that's too short

    def __call__(self, examples):
        call_type = {
            'sentence': self.process_sentence,
            'segment': self.process_segment,
            'note': self.process_note,
        }
        return call_type[self.process_type](examples)

    @staticmethod
    def _add_ids(new_ids, cls_id, sep_id):
        """add extra ids besides input_ids

        Args:
            new_ids (List): input_ids

        Return:
            tokenized dict with four ids
        """
        outdict = {
            'attention_mask': [], 'input_ids': [], 'special_tokens_mask': [], 'token_type_ids': []
        }
        for ids in new_ids:
            input_ids = [cls_id] + ids + [sep_id]
            attention_mask = [1 for _ in range(len(input_ids))]
            token_type_ids = [0 for _ in range(len(input_ids))]
            special_tokens_mask = [1] + [0 for _ in range(len(ids))] + [1]

            assert len(input_ids) == len(attention_mask) == len(token_type_ids) == len(special_tokens_mask)

            outdict['input_ids'].append(input_ids)
            outdict['attention_mask'].append(attention_mask)
            outdict['token_type_ids'].append(token_type_ids)
            outdict['special_tokens_mask'].append(special_tokens_mask)

        return outdict


    def process_sentence(self, examples):
        """
        Goal for sentence: remove too short
        """
        new_sentences = []
        for sent, sent_len in zip(examples['input_ids'], examples['length']):
            if sent_len >= self.MIN_SEQ_LEN:
                new_sentences.append(sent[:self.MAX_SEQ_LEN])

        new_examples = self._add_ids(new_sentences, cls_id=self.CLS_ID, sep_id=self.SEP_ID)
        return new_examples


    def process_segment(self, examples):
        """
        Goal for segment: group to form long-enough segment
        """
        new_segments= []
        running_seg = []
        for seg, seg_len in zip(examples['input_ids'], examples['length']):

            ## group short segs together
            if seg_len < self.MIN_SEQ_LEN:
                if seg_len >= self.MIN_SENT_LEN:
                    running_seg.extend(seg)
                if len(running_seg) > self.MIN_SEQ_LEN:
                    new_segments.append(running_seg)
                    running_seg = []
            else:
                new_segments.append(seg)
                # discard prev short segs to keep text in original order
                running_seg = []

        # truncate to max len
        new_input_ids = [seg[:self.MAX_SEQ_LEN] for seg in new_segments]
        new_examples = self._add_ids(new_input_ids, cls_id=self.CLS_ID, sep_id=self.SEP_ID)
        return new_examples


    def process_note(self, examples):
        """
        Goal for note: chunk long notes
        """
        
        note_chunks = []
        category_chunks = []

        for note, note_len, note_cat in zip(
            examples['input_ids'], examples['length'], examples['note_category']
        ):
            # remove short notes
            if note_len < self.MIN_SEQ_LEN:
                continue
            else:
                # chunk note by max seq len
                num_note = note_len // self.MAX_SEQ_LEN
                if num_note > 0:
                    for i in range(0, num_note):
                        note_chunks.append(
                            note[i*self.MAX_SEQ_LEN: (i+1)*self.MAX_SEQ_LEN]
                        )
                        category_chunks.append(note_cat)

                # see if remainder is long enough to be kept
                remainder = note_len % self.MAX_SEQ_LEN
                if remainder > self.MIN_SEQ_LEN:
                    note_chunks.append(note[-remainder:])
                    category_chunks.append(note_cat)

        new_examples = self._add_ids(note_chunks, cls_id=self.CLS_ID, sep_id=self.SEP_ID)
        new_examples['note_category'] = category_chunks

        return new_examples


def process_lines(raw_text_path, tokenizer, cache_dir,
        process_type, MIN_SEQ, MAX_SEQ, 
        N_JOBS=1, note_category=None, overwrite_cache=True):

    # load raw text
    ds = load_dataset('text', data_files=raw_text_path, cache_dir=cache_dir)['train']

    # tokenize 
    tok = TokenizeEngine(tokenizer)

    ts = ds.map(tok, batched=True, num_proc=N_JOBS,
        remove_columns=['text'], load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    # add note cat id 
    if process_type == 'note':
        assert note_category is not None

        lbl = LabelEngine(note_category)

        ts = ts.map(lbl, load_from_cache_file=False, desc="Adding note category ids")

    # process lines
    cls_id = tokenizer.vocab[tokenizer.special_tokens_map['cls_token']]
    sep_id = tokenizer.vocab[tokenizer.special_tokens_map['sep_token']]

    proc = ProcessEngine(
        process_type=process_type,
        MIN_SEQ_LEN=MIN_SEQ,
        MAX_SEQ_LEN=MAX_SEQ,
        CLS_ID=cls_id,
        SEP_ID=sep_id
    )

    ps = ts.map(proc, batched=True, num_proc=N_JOBS,
        remove_columns=['length'], load_from_cache_file=not overwrite_cache,
        desc="Processing lines",
    )

    return ps




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--RAW_TEXT_DIR", type=str, default='RAW_TEXT')
    parser.add_argument("--CACHE_DIR", type=str, default='tmp_cache')
    parser.add_argument("--PROCESSED_DIR", type=str, default='PROCESSED_FILES')

    parser.add_argument("--note_category", type=str, required=True)
    parser.add_argument("--process_type", type=str, default='sentence')

    parser.add_argument("--min_length", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--tokenizer_path", type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    parser.add_argument("--n_jobs", type=int, default=16)
    
    args = parser.parse_args()

    os.makedirs(args.PROCESSED_DIR, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path, cache_dir=args.CACHE_DIR)
    logging.info('Loaded tokenizer')

    raw_text_path = os.path.join(args.RAW_TEXT_DIR, args.note_category, f'{args.process_type}.txt')


    processed_dataset = process_lines(
        raw_text_path=raw_text_path,
        tokenizer=tokenizer,
        cache_dir=args.CACHE_DIR,
        process_type=args.process_type,
        MIN_SEQ=args.min_length,
        MAX_SEQ=args.max_length,
        N_JOBS=args.n_jobs,
        note_category=args.note_category
    )

    outname = f'{args.note_category}-{args.min_length}-{args.max_length}'
    processed_dataset.save_to_disk(
        os.path.join(args.PROCESSED_DIR, outname)
    )

    logging.info(f"Processed and saved {outname} ({args.process_type}) to disk")



if __name__ == "__main__":
    main()

    