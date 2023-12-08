
import numpy as np 
import pandas as pd 

ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']

def simple_imputer(df):
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]].copy()
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(0)
    
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.groupby(ID_COLS).cumsum() # a fix here
    hours_of_absence.columns.set_names(['LEVEL2', 'Aggregation Function'], inplace=True)
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].groupby(ID_COLS).fillna(method='ffill') #.fillna(100)
    
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out


def load_cohort(cohort_path):
    cohort = pd.read_pickle(cohort_path)

    if type(cohort) == dict:
        info_tr, info_dev, info_te = [cohort[s] for s in ['train', 'val', 'test']]

    elif type(cohort) == list:
        info_tr, info_dev, info_te = cohort 

    train_stay, dev_stay, test_stay = [df['ICUSTAY_ID'].values.tolist() for df in (info_tr, info_dev, info_te)]

    return [info_tr, info_dev, info_te], [train_stay, dev_stay, test_stay]

