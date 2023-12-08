import os, time, logging
import numpy as np 
import pandas as pd 

import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mimic_dir", default='~/physionet.org/files/mimiciii/1.4', type=str, help="Dir for MIMIC-III")
    parser.add_argument("--cohort_file", default='baseline_cohort.csv', type=str, help='CSV or pickle; requires SUBJECT_ID, HADM_ID, ICUSTAY_ID, INTIME, OUTTIME')
    parser.add_argument("--output_dir", default='hourly_104', type=str, help='output dir')

    parser.add_argument("--item_map_file", default='mimic_extract_resources/itemid_to_variable_map.csv', type=str)
    parser.add_argument("--var_range_file", default='mimic_extract_resources/variable_ranges.csv', type=str)

    parser.add_argument("--not_apply_var_limit", action='store_const', const=True, default=False)
    parser.add_argument("--debug", "-D", action='store_const', const=True, default=False)

    parser.add_argument("--max_hours", default=48, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)



    icu_df = load_cohort(args.cohort_file, args.debug)

    hr = args.max_hours
    raw_file = f'RAW_104_hourly_{hr}hr.csv'
    apply_var_limit = not args.not_apply_var_limit

    # load intermediate tmp file
    X_tmp_file = f'tmp_raw_{hr}hr.csv' if not args.debug else f'tmp_raw_{hr}hr_sample.csv'
    fpath = os.path.join(args.output_dir, X_tmp_file)
    if not os.path.isfile(fpath):
        # process raw
        X = process_raw(args.mimic_dir, icu_df, args.item_map_file)
        X.to_csv(fpath, index=False)
        logging.info('Saved intermediate file')

    else:
        X = pd.read_csv(fpath)
        logging.info('Loaded intermediate saved file')


    # debug 
    if args.debug:
        X = X.iloc[: 10000]
        raw_file = raw_file.replace('.csv', '_sample.csv')

    # filter data
    X_filtered = filter_raw(X, args.max_hours, args.mimic_dir, icu_df, args.item_map_file, args.var_range_file, apply_var_limit)

    # save data
    X_filtered.to_csv(os.path.join(args.output_dir, raw_file))

    logging.info('Finished')


def load_cohort(cohort_path, debug=False):

    _, ext = os.path.splitext(cohort_path)
    if ext == '.csv':
        cohort_df = pd.read_csv(cohort_path, usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID','INTIME','OUTTIME'])

    elif ext == '.p':
        cohort  = pd.read_pickle(cohort_path)
        if type(cohort) is dict:
            print(cohort.keys())
            cohort_df = pd.concat(list(cohort.values()))
        else:
            raise NotImplementedError

    print(len(cohort_df), 'stays')
    if debug:
        cohort_df = cohort_df.head(100)

    return cohort_df

def process_raw(MIMIC_DIR, icu_df, mimic_mapping_filename):
    # no selection criteria for now, just retrieve everything for the icustays
    

    var_map = get_variable_mapping(mimic_mapping_filename)
    # select itemid, etc
    hadms_to_keep = set([ float(i) for i in icu_df['HADM_ID']])
    icustays_to_keep = set([ float(i) for i in icu_df['ICUSTAY_ID']])

    chartitems_to_keep = var_map.loc[var_map['LINKSTO'] == 'chartevents'].ITEMID
    chartitems_to_keep = set([ int(i) for i in chartitems_to_keep ])

    labitems_to_keep = var_map.loc[var_map['LINKSTO'] == 'labevents'].ITEMID
    labitems_to_keep = set([ int(i) for i in labitems_to_keep ])


    # LOAD LABEVENTS
    logging.info('Loading and processing lab events')
    lab_cols = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE',
           'VALUENUM', 'VALUEUOM']
    lab_iter = pd.read_csv('%s/LABEVENTS.csv.gz' % MIMIC_DIR, usecols=lab_cols,
                       #nrows=1e+6, 
                       iterator=True, chunksize=200000)
    lab_list = []
    for d in lab_iter:
        lab_list.append(d[ (d['ITEMID'].isin(labitems_to_keep)) & (d['HADM_ID'].isin(hadms_to_keep))])
    labs = pd.concat(lab_list)
    logging.info(f'raw lab rows: {len(labs)}')

    # raw filtering labs to match original query
    labs['HADM_ID'] = labs['HADM_ID'].astype(int)
    df_lbs = pd.merge(icu_df, labs, on=['SUBJECT_ID', 'HADM_ID'], how='right')
    mask1 = pd.to_datetime(df_lbs['CHARTTIME']) + pd.Timedelta(6, unit='h') >= pd.to_datetime(df_lbs['INTIME'])
    mask2 = pd.to_datetime(df_lbs['OUTTIME']) >= pd.to_datetime(df_lbs['CHARTTIME']) # >= instead of >
    mask3 = df_lbs['VALUENUM'] > 0
    df_lbs = df_lbs[mask1 & mask2 & mask3].drop_duplicates()
    logging.info(f'filtered lab rows: {len(df_lbs)}')



    # LOAD CHARTEVENTS
    logging.info('Loading and processing chart events')
    chart_cols = ['ICUSTAY_ID', 'ITEMID', 
            'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'ERROR']
    chart_iter = pd.read_csv('%s/CHARTEVENTS.csv.gz' % MIMIC_DIR, usecols=chart_cols,
                       #nrows=1e+6, 
                       iterator=True, chunksize=200000)
    chart_list = []
    for d in chart_iter:
        chart_list.append(d[ (d['ITEMID'].isin(chartitems_to_keep)) & (d['ICUSTAY_ID'].isin(icustays_to_keep))])
    charts = pd.concat(chart_list)
    logging.info(f'raw chart rows: {len(charts)}')

    # raw filtering charts to match orignical query
    charts['ICUSTAY_ID'] = charts['ICUSTAY_ID'].astype(int)
    icu_tmp = icu_df[['ICUSTAY_ID','INTIME','OUTTIME']]
    df_chts = pd.merge(icu_tmp, charts, on=['ICUSTAY_ID'], how='right')

    mask1 = pd.to_datetime(df_chts['CHARTTIME']) >= pd.to_datetime(df_chts['INTIME'])
    mask2 = pd.to_datetime(df_chts['OUTTIME']) >= pd.to_datetime(df_chts['CHARTTIME']) 
    mask3 = df_chts['ERROR'] != 1
    mask4 = df_chts['VALUENUM'].notnull()
    df_chts = df_chts[mask1 & mask2 & mask3 & mask4]

    # a test case: tmp should contain four records
    # tmp=df_chts[ (df_chts.ITEMID == 833) & (df_chts.ICUSTAY_ID ==235907) ]
    # logging.info(tmp)
    # logging.info(tmp.CHARTTIME)
    logging.info(f'filtered chart rows: {len(df_chts)}')
    df_chts = pd.merge(icu_df, df_chts, on=['ICUSTAY_ID','INTIME','OUTTIME'], how='right')
    # logging.info('merged cols', df_chts.columns)
    logging.info(f'merged hadm chart rows: {len(df_chts)}')
    df_chts.dropna()
    logging.info(f'dropna chart rows: {len(df_chts)}')



    # COMBINE THE TWO 
    shared_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',  
            'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'INTIME']
    X = pd.concat([df_lbs[shared_cols], df_chts[shared_cols]])
    return X

def filter_raw(X, MAX_HOURS, MIMIC_DIR, icu_df, 
                    mimic_mapping_filename, range_filename, apply_var_limit=True):
    """
    X: raw output from process_raw(), contain raw charts and labs 
        curated based on mimic mapping file, for cohort icu stays 
        based on intime, outtime, charttime, etc

    return: new X with values clipped based on ranges, 
            unit standardization, value aggregation, 
            and time-series construction
    """
    var_ranges = get_variable_ranges(range_filename)
    var_map = get_variable_mapping(mimic_mapping_filename)
    var_map = var_map[['LEVEL2', 'ITEMID', 'LEVEL1']]


    icu_df = icu_df.reset_index().set_index('ICUSTAY_ID')


    # load d_items
    itemids = set(X.ITEMID)
    d1 = pd.read_csv('%s/D_ITEMS.csv.gz' % MIMIC_DIR, usecols=['ITEMID', 'LABEL'])
    d2 = pd.read_csv('%s/D_LABITEMS.csv.gz' % MIMIC_DIR, usecols=['ITEMID', 'LABEL'])
    I = pd.concat([d1,d2])
    I = I[I.ITEMID.isin(itemids)] #.set_index('ITEMID') 
    logging.info(f'counts of d_items + d_labitems for cohort: {len(I)}')


    # prepare X
    to_hours = lambda x: max(0, x.days*24 + x.seconds // 3600)

    ID_COLS = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']
    
    for k in ID_COLS:
        X[k] = X[k].astype(int)
    X['VALUE'] = pd.to_numeric(X['VALUE'], 'coerce')
    X['hours_in'] = (pd.to_datetime(X['CHARTTIME']) - pd.to_datetime(X['INTIME'])).apply(to_hours)
    logging.info('joining vars')

    X = pd.merge(X, var_map, on=['ITEMID'])
    X = pd.merge(X, I, on=['ITEMID'])
    X.set_index('ICUSTAY_ID', inplace=True)
    X.set_index('ITEMID', append=True, inplace=True)


    # standardize units
    X.drop(columns=['CHARTTIME', 'INTIME'], inplace=True)
    logging.info('STANDARDIZING')
    standardize_units(X, name_col = 'LEVEL1', inplace=True)

    X.set_index(['LABEL', 'LEVEL2', 'LEVEL1'], append=True, inplace=True)
    # print(X.columns, X.index.names)


    # apply var limit
    logging.info('APPLYING VAR LIMIT')
    if apply_var_limit:
        X = apply_variable_limits(X, var_ranges, 'LEVEL2')

    # agg values
    logging.info('GROUPING BY')
    X = X.groupby(ID_COLS+['LEVEL2', 'hours_in']).agg(['mean', 'std', 'count'])
    X.columns = X.columns.droplevel(0)
    X.columns.names = ['Aggregation Function']



    # construct ts
    logging.info('CREATING NEW INDEX W HOURLY')
    icu_df['max_hours'] = ( pd.to_datetime(icu_df['OUTTIME']) -  pd.to_datetime(icu_df['INTIME']) ).apply(to_hours)
    hour_indexer = icu_df['max_hours'] > MAX_HOURS
    icu_df.loc[hour_indexer, 'max_hours']= MAX_HOURS
    # unnest df
    missing_hours_fill = range_unnest(icu_df, 'max_hours', 
                        out_col_name='hours_in', reset_index=True)
    missing_hours_fill['tmp'] = np.NaN
    fill_df = icu_df.reset_index()[ID_COLS].join(missing_hours_fill.set_index('ICUSTAY_ID'), on='ICUSTAY_ID')
    fill_df.set_index(ID_COLS+['hours_in'], inplace=True)

    X = X.unstack(level = ['LEVEL2'])
    X.columns = X.columns.reorder_levels(order=['LEVEL2', 'Aggregation Function'])
    # fill to the index
    X = X.reindex(fill_df.index)
    X = X.sort_index(axis=0).sort_index(axis=1)



    # fix nan count
    idx = pd.IndexSlice
    X.loc[:, idx[:, 'count']] = X.loc[:, idx[:, 'count']].fillna(0)
    logging.info(f"shape of X: {X.shape}")

    return X





##############
# helpers
##############

def range_unnest(df, col, out_col_name=None, reset_index=False):
    assert len(df.index.names) == 1, "Does not support multi-index."
    if out_col_name is None: out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(y+1)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index: col_flat = col_flat.set_index(df.index.names[0])
    return col_flat

def apply_variable_limits(df, var_ranges, var_names_index_col='LEVEL2'):
    idx_vals        = df.index.get_level_values(var_names_index_col)
    non_null_idx    = ~df.VALUE.isnull()
    var_names       = set(idx_vals)
    var_range_names = set(var_ranges.index.values)

    for var_name in var_names:
        var_name_lower = var_name.lower()
        if var_name_lower not in var_range_names:
            print("No known ranges for %s" % var_name)
            continue

        outlier_low_val, outlier_high_val, valid_low_val, valid_high_val = [
            var_ranges.loc[var_name_lower, x] for x in ('OUTLIER_LOW','OUTLIER_HIGH','VALID_LOW','VALID_HIGH')
        ]

        running_idx = non_null_idx & (idx_vals == var_name)

        outlier_low_idx  = (df.VALUE < outlier_low_val)
        outlier_high_idx = (df.VALUE > outlier_high_val)
        valid_low_idx    = ~outlier_low_idx & (df.VALUE < valid_low_val)
        valid_high_idx   = ~outlier_high_idx & (df.VALUE > valid_high_val)

        var_outlier_idx   = running_idx & (outlier_low_idx | outlier_high_idx)
        var_valid_low_idx = running_idx & valid_low_idx
        var_valid_high_idx = running_idx & valid_high_idx

        df.loc[var_outlier_idx, 'VALUE'] = np.nan
        df.loc[var_valid_low_idx, 'VALUE'] = valid_low_val
        df.loc[var_valid_high_idx, 'VALUE'] = valid_high_val

        n_outlier = sum(var_outlier_idx)
        n_valid_low = sum(var_valid_low_idx)
        n_valid_high = sum(var_valid_high_idx)
        if n_outlier + n_valid_low + n_valid_high > 0:
            print(
                "%s had %d / %d rows cleaned:\n"
                "  %d rows were strict outliers, set to np.nan\n"
                "  %d rows were low valid outliers, set to %.2f\n"
                "  %d rows were high valid outliers, set to %.2f\n"
                "" % (
                    var_name,
                    n_outlier + n_valid_low + n_valid_high, sum(running_idx),
                    n_outlier, n_valid_low, valid_low_val, n_valid_high, valid_high_val
                )
            )

    return df


def get_values_by_name_from_df_column_or_index(data_df, colname):
    """ Easily get values for named field, whether a column or an index

    Returns
    -------
    values : 1D array
    """
    try:
        values = data_df[colname]
    except KeyError as e:
        if colname in data_df.index.names:
            values = data_df.index.get_level_values(colname)
        else:
            raise e
    return values

UNIT_CONVERSIONS = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),
    ('fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),
    ('oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),
    ('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),
    ('height',                   'in',  None,             lambda x: x*2.54),
]

def standardize_units(X, name_col='ITEMID', unit_col='VALUEUOM', value_col='VALUE', inplace=True):
    if not inplace: X = X.copy()
    name_col_vals = get_values_by_name_from_df_column_or_index(X, name_col)
    unit_col_vals = get_values_by_name_from_df_column_or_index(X, unit_col)

    try:
        name_col_vals = name_col_vals.str
        unit_col_vals = unit_col_vals.str
    except:
        print("Can't call *.str")
        print(name_col_vals)
        print(unit_col_vals)
        raise

    #name_filter, unit_filter = [
    #    (lambda n: col.contains(n, case=False, na=False)) for col in (name_col_vals, unit_col_vals)
    #]
    # TODO(mmd): Why does the above not work, but the below does?
    name_filter = lambda n: name_col_vals.contains(n, case=False, na=False)
    unit_filter = lambda n: unit_col_vals.contains(n, case=False, na=False)

    for name, unit, rng_check_fn, convert_fn in UNIT_CONVERSIONS:
        name_filter_idx = name_filter(name)
        needs_conversion_filter_idx = name_filter_idx & False

        if unit is not None: needs_conversion_filter_idx |= name_filter(unit) | unit_filter(unit)
        if rng_check_fn is not None: needs_conversion_filter_idx |= rng_check_fn(X[value_col])

        idx = name_filter_idx & needs_conversion_filter_idx

        X.loc[idx, value_col] = convert_fn(X[value_col][idx])

    return X

def get_variable_mapping(mimic_mapping_filename):
    # Read in the second level mapping of the itemids
    var_map = pd.read_csv(mimic_mapping_filename, index_col=None)
    var_map = var_map[(var_map['LEVEL2'] != '') & (var_map['COUNT']>0)]
    var_map = var_map[(var_map['STATUS'] == 'ready')]
    var_map['ITEMID'] = var_map['ITEMID'].astype(int)

    return var_map

def get_variable_ranges(range_filename):
    # Read in the second level mapping of the itemid, and take those values out
    columns = [ 'LEVEL2', 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH' ]
    to_rename = dict(zip(columns, [ c.replace(' ', '_') for c in columns ]))
    to_rename['LEVEL2'] = 'VARIABLE'
    var_ranges = pd.read_csv(range_filename, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(columns=to_rename, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges['VARIABLE'] = var_ranges['VARIABLE'].str.lower()
    var_ranges.set_index('VARIABLE', inplace=True)
    var_ranges = var_ranges.loc[var_ranges.notnull().all(axis=1)]

    return var_ranges


if __name__ == '__main__':
    main()