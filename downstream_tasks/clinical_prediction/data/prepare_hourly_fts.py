import os, logging
import numpy as np 
import pandas as pd 
import pickle as pk 

import argparse

from tqdm import tqdm
from utils import load_cohort, simple_imputer

'''
Preprocessing follows MIMIC-Extract for the hourly features, which involves normalization and simple forward-fill imputation, etc. 
The script saves train/dev/test files that should be ready for modeling.
But it also provides function to create split on the fly when given new, predefined splits.
'''

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_file", default='split.p', type=str, help='pickled file with defined train/dev/test')

    parser.add_argument("--output_dir", default='hourly_104', type=str, help='output dir')
    parser.add_argument("--raw_file", default='RAW_104_hourly_48hr.csv', type=str, help='raw file path')
    parser.add_argument("--out_name", default='input_104_hourly_df.p', type=str, help='save file name')

    parser.add_argument("--window_size", default=24, type=int)
    args = parser.parse_args()

    # load cohort
    cohort_dfs, cohort_stays = load_cohort(args.cohort_file)
    logging.info("cohort loaded in train/val/test")

    # load raw hourly data
    logging.info("reading input data")
    df = pd.read_csv(os.path.join(args.output_dir, args.raw_file), header=[0,1], index_col=[0,1,2,3])
    df.rename_axis(index=str.lower, inplace=True)

    # prepare
    logging.info("processing input data")
    df_train, df_dev, df_test = prepare_data_with_splits(df, args.window_size, *cohort_stays)

    with open(os.path.join(args.output_dir, args.out_name), 'wb') as outf:
        pk.dump([df_train, df_dev, df_test], outf)

    # # convert to array
    # input_array_list = [_load_array_data(info, df) 
    #                 for info, df in zip(cohort_dfs, [df_train, df_dev, df_test])]   

    # with open(os.path.join(args.output_dir, 'input_104_hourly_array.p'), 'wb') as outf:
    #     pk.dump(input_array_list, outf)  


def prepare_data_with_splits(df, WINDOW_SIZE, train_stay, dev_stay, test_stay):
    '''
    stays: icustay_id
    raw_data: df loaded from raw file
    window_size: input hours

    return: df_train, df_dev, df_test (input for modeling)
    '''

    df_train = df[df.index.get_level_values('icustay_id').isin(train_stay)]
    df_dev = df[df.index.get_level_values('icustay_id').isin(dev_stay)]
    df_test = df[df.index.get_level_values('icustay_id').isin(test_stay)]

    logging.info("PREPROCESSING")
    # unnest data to align hours
    df_train = unnest_df(df_train, WINDOW_SIZE)
    df_dev = unnest_df(df_dev, WINDOW_SIZE)
    df_test = unnest_df(df_test, WINDOW_SIZE)

    # standardization
    logging.info("STANDARDIZATION")
    idx = pd.IndexSlice
    tr_means, tr_stds = df_train.loc[:, idx[:,'mean']].mean(axis=0), df_train.loc[:, idx[:,'mean']].std(axis=0)

    df_train.loc[:, idx[:,'mean']] = (df_train.loc[:, idx[:,'mean']] - tr_means)/tr_stds
    df_dev.loc[:, idx[:,'mean']] = (df_dev.loc[:, idx[:,'mean']] - tr_means)/tr_stds
    df_test.loc[:, idx[:,'mean']] = (df_test.loc[:, idx[:,'mean']] - tr_means)/tr_stds

    # imputation
    logging.info("IMPUTATION")
    # global_means = df_train.loc[:, idx[:, 'mean']].mean(axis=0)
    df_train, df_dev, df_test = [simple_imputer(df) for df in (df_train, df_dev, df_test)]
    logging.info("data ready")

    return df_train, df_dev, df_test


def _load_array_data(info, df):
    logging.info(f'loading data into arrays: {len(info)}')
    idx = pd.IndexSlice
    df = df.loc[:, idx[:, ['mask', 'mean']]]

    X = []

    info.reset_index(inplace=True)
    for n, row in tqdm(info.iterrows(), total=len(info)):
        subj = row['SUBJECT_ID']
        hadm = row['HADM_ID']
        icu  = row['ICUSTAY_ID']

        msk = _id_msk(df, subj, hadm, icu)
        ts = df[msk].values

        X.append(ts)

    X = np.array(X)
    return X

def _id_msk(df, subj, hadm, icu):
    msk1 = df.index.get_level_values('subject_id') == subj
    msk2 = df.index.get_level_values('hadm_id') == hadm
    msk3 = df.index.get_level_values('icustay_id') == icu
    return msk1 & msk2 & msk3


def unnest_df(value_df, hour=24):
    IDS = pd.Series([(a,b,c) for a,b,c,d in value_df.index.tolist()]).unique()
    ID_COLS = ['SUBJECT_ID'.lower(), 'HADM_ID'.lower(), 'ICUSTAY_ID'.lower()]
    
    icu_df = pd.DataFrame.from_records(IDS, columns=ID_COLS)
    icu_df['tmp']=10
    return unnest_visit(icu_df, value_df, hour=hour)

def unnest_visit(icu_df, value_df, hour=24):
    ID_COLS = ['SUBJECT_ID'.lower(), 'HADM_ID'.lower(), 'ICUSTAY_ID'.lower()]
    icu_df=icu_df.set_index('ICUSTAY_ID'.lower())
    
    missing_hours_fill =range_unnest_hour(icu_df, 'tmp', hour=hour, out_col_name='hours_in')
    missing_hours_fill['tmp'] = np.NaN
    
    fill_df = icu_df.reset_index()[ID_COLS].join(missing_hours_fill, on='icustay_id')
    fill_df.set_index(ID_COLS+['hours_in'], inplace=True)
    
    idx=pd.IndexSlice
    new_df = value_df.reindex(fill_df.index).copy()
    new_df.loc[:, idx[:, 'count']] = new_df.loc[:, idx[:, 'count']].fillna(0)
    return new_df

def range_unnest_hour(df, col, hour=24, out_col_name='hours_in', reset_index=False):
    if out_col_name is None: out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(hour)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index: col_flat = col_flat.set_index(df.index.names[0])
    return col_flat



if __name__ == '__main__':
    main()

