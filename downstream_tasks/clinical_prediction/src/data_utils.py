import numpy as np 
import pandas as pd 
from tqdm import tqdm

import torch, os

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class ExpDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        # task/input config
        self.target = args.task.upper() # e.g., MORT_HOSP
        self.modality = args.modality
        self.batch_size = args.batch_size
        self.silent = args.silent

        self.note_encode_dir = args.note_encode_dir
        self.note_encode_name = args.note_encode_name
        
        root_dir = os.path.abspath(args.root_dir)

        # cohort df 
        cohort_name = 'mort_los' if self.target != 'DRG' else 'drg'
        all_cohort_df = pd.read_pickle(os.path.join(root_dir, 'cohorts', f'{cohort_name}.p'))
        self.train_df, self.val_df, self.test_df = all_cohort_df['train'], all_cohort_df['val'], all_cohort_df['test']
        if args.debug:
            self.train_df, self.val_df, self.test_df = [df.head(200) for df in [self.train_df, self.val_df, self.test_df]]


        # data df 
        if self.modality == 'struct':
            self.my_collate_fn = None 

            self.train_data_ts, self.val_data_ts, self.test_data_ts = pd.read_pickle(os.path.join(root_dir, 'data_structured', f'{cohort_name}_104_hourly_df.p'))

            self.data_ts = [self.train_data_ts, self.val_data_ts, self.test_data_ts]
            self.X_mean = get_X_mean(self.train_data_ts)

        elif self.modality == 'text':
            self.my_collate_fn = collate_notes

            self.data_note = pd.read_pickle(os.path.join(root_dir, self.note_encode_dir, f'{self.note_encode_name}-{cohort_name}.p'))


        elif self.modality == 'both':
            self.my_collate_fn = collate_both

            self.data_note = pd.read_pickle(os.path.join(root_dir, self.note_encode_dir, f'{self.note_encode_name}-{cohort_name}.p'))
            
            self.data_ts = pd.read_pickle(os.path.join(root_dir, 'data_structured', f'{cohort_name}_104_hourly_df.p'))

            self.X_mean = get_X_mean(self.data_ts[0])
        else:
            raise NotImplementedError


    def init_datasets(self, cohort_df_list):

        if self.modality == 'struct':
            dataset_list = [
                StructDataset(self.target, cohort_df, data_df, self.silent)
                for cohort_df, data_df in zip(cohort_df_list, self.data_ts)
            ] 

        elif self.modality == 'text':
            dataset_list = [
                TextDataset(self.target, cohort_df, self.data_note, self.silent)
                for cohort_df in cohort_df_list
            ]

        elif self.modality == 'both':
            dataset_list = [
                BimodalDataset(self.target, cohort_df, [self.data_note, data_df], self.silent)
                for cohort_df, data_df in zip(cohort_df_list, self.data_ts)
            ]

        else:
            raise NotImplementedError

        return dataset_list


    def setup(self, stage=None):
        if stage == None:
            self.train_dataset, self.val_dataset, self.test_dataset = self.init_datasets(
                [self.train_df, self.val_df, self.test_df]
            )

        elif stage == 'test':
            self.test_dataset = self.init_datasets([self.test_df])

    def train_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.my_collate_fn)
    def val_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate_fn)
    def test_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate_fn)



def collate_notes(batch):

    notes = [ torch.tensor(np.array(b[0])) for b in batch ]
    notes = pad_sequence(notes, batch_first=True)

    labels = torch.tensor(np.array([ b[-2] for b in batch ]))
    stays  = np.array([ b[-1] for b in batch ])

    return notes, labels, stays

def collate_both(batch):

    notes = [ torch.tensor(np.array(b[0][0])) for b in batch ]
    notes = pad_sequence(notes, batch_first=True)

    struct = torch.tensor(np.array([b[0][1] for b in batch]))

    labels = torch.tensor(np.array([ b[-2] for b in batch ]))
    stays  = np.array([ b[-1] for b in batch ])

    return (notes, struct), labels, stays


class TemplateDataset(Dataset):
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class StructDataset(TemplateDataset):
    def __init__(self, target, cohort_df, ts_df, silent=False):
        super().__init__()

        self.data = []

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):
            subj, hadm, icu = row['SUBJECT_ID'], row['HADM_ID'], row['ICUSTAY_ID']


            msk = id_msk(ts_df, subj, hadm, icu)
            
            x = ts_df[msk].values
            y = row[target]

            self.data.append([x, y, hadm])

    
class TextDataset(TemplateDataset):
    def __init__(self, target, cohort_df, note_df, silent=False):
        super().__init__()

        self.data = []

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):
            hadm = row['HADM_ID']

            x = note_df[note_df['HADM_ID'] == hadm]['vector'].tolist()

            y = row[target]

            self.data.append([x, y, hadm])


class BimodalDataset(TemplateDataset):
    def __init__(self, target, cohort_df, data_dfs, silent=False):
        super().__init__()

        assert len(data_dfs) == 2
        note_df, ts_df = data_dfs

        self.data = []

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):

            subj, hadm, icu = row['SUBJECT_ID'], row['HADM_ID'], row['ICUSTAY_ID']
            
            msk = id_msk(ts_df, subj, hadm, icu)
            x_ts = ts_df[msk].values

            x_txt = note_df[note_df['HADM_ID'] == hadm]['vector'].tolist()

            y = row[target]

            self.data.append([(x_txt, x_ts), y, hadm])


def id_msk(df, subj, hadm, icu):
    msk1 = df.index.get_level_values('subject_id') == subj
    msk2 = df.index.get_level_values('hadm_id') == hadm
    msk3 = df.index.get_level_values('icustay_id') == icu
    return msk1 & msk2 & msk3

def to_3D_tensor(df):
    idx = pd.IndexSlice
    return np.dstack([df.loc[idx[:,:,:,i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))])

def get_X_mean(lvl2_train):
    X_mean = np.nanmean(
            to_3D_tensor(
                lvl2_train.loc[:, pd.IndexSlice[:, 'mean']] * 
                np.where((lvl2_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN)
            ),
            axis=0, keepdims=True
        ).transpose([0, 2, 1])
    X_mean = np.nan_to_num(X_mean,0)
    return X_mean