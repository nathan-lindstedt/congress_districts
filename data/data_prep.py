#%%
# Import libraries
import os

import pandas as pd
import re

from contextlib import contextmanager
from logging import basicConfig, exception, ERROR

#%%
# System path variables
start: str = os.path.dirname(__file__)

#%%
# Context manager
@contextmanager
def incoming(input_path):
    """[Opens read-only .txt file with encoding='Windows-1252']
    
    Arguments:
        input_path {[str]} -- [Input path]
    
    Yields:
        [obj] -- [Read-only .txt file with encoding='Windows-1252']
    """
    try:
        infile = open(input_path, 'r', encoding='Windows-1252') 
    except FileNotFoundError as incoming_error:
        exception(incoming_error)
    except Exception as incoming_e:
        exception(incoming_e)
    else:
        yield infile
    finally:
        infile.close()

#%%
# Data preparation functions
def get_source(line):
    source_line = re.compile(r'(Source code:)(\s*)(.*)')
    source_search = source_line.search(line)

    if source_search:
        source_code = source_search.group(3)
    
    return source_code

def get_nhgis(line):
    nhgis_line = re.compile(r'(NHGIS code:)(\s*)(.*)')
    nhgis_search = nhgis_line.search(line)

    if nhgis_search:
        nhgis_code = nhgis_search.group(3)

    return nhgis_code

def get_dict(input_path):
    dict_cd = {}
    source_code, nhgis_code = None, None

    with incoming(input_path) as infile:
        line = infile.readline()
        
        while line:
            line = infile.readline()

            try:
                source_code = get_source(line)
            except:
                pass

            try:
                nhgis_code = get_nhgis(line)
            except:
                pass

            if None not in (source_code, nhgis_code):
                dict_cd[nhgis_code] = source_code
                source_code, nhgis_code = None, None
    
    return dict_cd

def rename_df_columns(X_df, dict_cd):
    rename_list = list(X_df.columns)

    for index, data in enumerate(rename_list):
        for key, _ in dict_cd.items():
            if key in data:
                rename_list[index]=data.replace(key, dict_cd[key])

    return X_df.set_axis(rename_list, axis=1).dropna(axis=1)


#%%
# Load the data
X_train = pd.read_csv(os.path.relpath(f'nhgis0002_ds249_20205_cd116th.csv', start=start), low_memory=False)
X_test = pd.read_csv(os.path.relpath(f'nhgis0001_ds262_20225_cd118th.csv', start=start), low_memory=False)
y = pd.read_csv(os.path.relpath(f'house_outcomes_historical.csv', start=start), low_memory=False)

#%%
# Get source and NHGIS code
dict_cd116 = get_dict(os.path.relpath(f'nhgis0002_ds249_20205_cd116th_codebook.txt', start=start))
dict_cd118 = get_dict(os.path.relpath(f'nhgis0001_ds262_20225_cd118th_codebook.txt', start=start))

#%%
# Rename columns
X_train = rename_df_columns(X_train, dict_cd116)
X_test = rename_df_columns(X_test, dict_cd118)

#%%
# Keep common columns
common_cols = [col for col in set(X_train.columns).intersection(X_test.columns)]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

#%%
# Preprocessing
y['GISJOIN'] = 'G' + y['state_fips'].astype(str).str.zfill(2) + y['district'].astype(str).str.zfill(3)
y_train = y[(y['year'] == 2020) & (y['candidatevotes'] == y.groupby(['GISJOIN', 'year'])['candidatevotes'].transform('max'))]
y_test = y[(y['year'] == 2022) & (y['candidatevotes'] == y.groupby(['GISJOIN', 'year'])['candidatevotes'].transform('max'))]
y_train = pd.get_dummies(y_train, prefix=['party'], columns=['party'])
y_test = pd.get_dummies(y_test, prefix=['party'], columns=['party'])
y_train.drop(y_train[y_train['totalvotes'] <= 0].index, inplace=True)
y_test.drop(y_test[y_test['totalvotes'] <= 0].index, inplace=True)
train_set = y_train.merge(X_train, on='GISJOIN')
test_set = y_test.merge(X_test, on='GISJOIN')

#%%
# Export data
train_set.to_csv(os.path.relpath(f'../output/train_set.csv', start=start), index=False)
test_set.to_csv(os.path.relpath(f'../output/test_set.csv', start=start), index=False)

#%%