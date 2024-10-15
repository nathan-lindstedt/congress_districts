#%%
# Import libraries
import os

import pandas as pd
import re

from contextlib import contextmanager
from logging import exception

from sklearn.decomposition import TruncatedSVD

#%%
# System path variables
start: str = os.path.dirname(__file__)

#%%
# Global variables
context_fields = ['GISJOIN','YEAR','STUSAB',
                  'REGIONA','DIVISIONA','STATE',
                  'STATEA','COUNTYA','COUSUBA',
                  'PLACEA','TRACTA','BLKGRPA',
                  'CONCITA','AIANHHA','RES_ONLYA',
                  'TRUSTA','AIHHTLI','AITSCEA',
                  'ANRCA','CBSAA','CSAA',
                  'METDIVA','NECTAA','CNECTAA',
                  'NECTADIVA','UAA','CDCURRA',
                  'SLDUA','SLDLA','ZCTA5A',
                  'SUBMCDA','SDELMA','SDSECA',
                  'SDUNIA','PCI','PUMAA',
                  'GEOID','BTTRA','BTBGA',
                  'NAME_E','GEO_ID','TL_GEO_ID',
                  'NAME_M','AITSA']

#%%
# Data preparation functions
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
    column_list = list(X_df.columns)
    rename_list = []

    for index, data in enumerate(column_list):
        for key, _ in dict_cd.items():
            if key in data:
                column_list[index]=data.replace(key, dict_cd[key])
                rename_list.append(column_list[index])

    return rename_list, X_df.set_axis(column_list, axis=1).dropna(axis=1)


#%%
# Load the data
X_train = pd.read_csv(os.path.relpath(f'nhgis0002_ds249_20205_cd116th.csv', start=start), low_memory=False)
X_test = pd.read_csv(os.path.relpath(f'nhgis0001_ds262_20225_cd118th.csv', start=start), low_memory=False)
y = pd.read_csv(os.path.relpath(f'house_outcomes_historical.csv', start=start), low_memory=False)
poll = pd.read_csv(os.path.relpath(f'538_house_polls_historical.csv', start=start), low_memory=False)

#%%
# Get source and NHGIS code
dict_cd116 = get_dict(os.path.relpath(f'nhgis0002_ds249_20205_cd116th_codebook.txt', start=start))
dict_cd118 = get_dict(os.path.relpath(f'nhgis0001_ds262_20225_cd118th_codebook.txt', start=start))

#%%
# Rename columns
X_train_gisjoin = pd.DataFrame(X_train['GISJOIN'])
X_test_gisjoin = pd.DataFrame(X_test['GISJOIN'])
X_train_cols = [col for col in X_train.columns if col not in context_fields]
X_test_cols = [col for col in X_test.columns if col not in context_fields]
X_train_rename, X_train = rename_df_columns(X_train[X_train_cols], dict_cd116)
X_test_rename, X_test = rename_df_columns(X_test[X_test_cols], dict_cd118)

#%%
# Keep common columns
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

#%%
# Truncated SVD w/ automatic component selection for dimensionality reduction
svd_performance = []
base = X_train.shape[0]//10
scree = range(1, X_train.shape[1]//base, 1)

for n in scree:
    scree_svd = TruncatedSVD(n_components=n, random_state=42) 
    scree_fit = scree_svd.fit(X_train[common_cols])
    svd_performance.append(scree_fit.explained_variance_ratio_.sum())

n_components = len([x for x in svd_performance if x <= .95])
model_svd = TruncatedSVD(n_components=n_components)
X_train_svd = pd.DataFrame(model_svd.fit_transform(X_train[common_cols]))
X_test_svd = pd.DataFrame(model_svd.transform(X_test[common_cols]))

#%%
# Preprocessing
X_train = pd.concat([X_train_gisjoin, X_train_svd], axis=1)
X_test = pd.concat([X_test_gisjoin, X_test_svd], axis=1)
y['GISJOIN'] = 'G' + y['state_fips'].astype(str).str.zfill(2) + y['district'].astype(str).str.zfill(3)
y_train = y[(y['year'] == 2020) & (y['candidatevotes'] == y.groupby(['GISJOIN', 'year'])['candidatevotes'].transform('max'))]
y_test = y[(y['year'] == 2022) & (y['candidatevotes'] == y.groupby(['GISJOIN', 'year'])['candidatevotes'].transform('max'))]
y_train = pd.get_dummies(y_train, prefix=['party'], columns=['party'])
y_test = pd.get_dummies(y_test, prefix=['party'], columns=['party'])
poll_train = poll[(poll['cycle'] == 2020)]
                #   & (poll['candidatevotes'] == poll.groupby(['GISJOIN', 'year'])['candidatevotes'].transform('max'))]
poll_test = poll[(poll['cycle'] == 2022)]
                #   & (poll['candidatevotes'] == poll.groupby(['GISJOIN', 'year'])['candidatevotes'].transform('max'))]
y_train.drop(y_train[y_train['totalvotes'] <= 0].index, inplace=True)
y_test.drop(y_test[y_test['totalvotes'] <= 0].index, inplace=True)
train_set = y_train.merge(X_train, on='GISJOIN')
test_set = y_test.merge(X_test, on='GISJOIN')

#%%
# Export data
train_set.to_csv(os.path.relpath(f'../output/train_set.csv', start=start), index=False)
test_set.to_csv(os.path.relpath(f'../output/test_set.csv', start=start), index=False)

#%%