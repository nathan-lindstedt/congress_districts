#%%
# Import libraries
import os

import pandas as pd
import re

from contextlib import contextmanager
from logging import exception
from typing import Dict, List

from sklearn.decomposition import TruncatedSVD

#%%
# System path variables
start: str = os.path.dirname(__file__)

#%%
# Global variables
context_fields: List = ['GISJOIN','YEAR','STUSAB',
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

party_cols: List = ['REP', 'DEM']

#%%
# Data preparation functions
@contextmanager
def incoming(input_path: str) -> object:
    """
    Context manager for opening a file with the specified input path.

    Args:
        input_path (str): The path to the file to be opened.

    Yields:
        object: The file object opened in read mode with 'Windows-1252' encoding.

    Raises:
        FileNotFoundError: If the file specified by input_path does not exist.
        Exception: For any other exceptions that occur during file opening.

    Example:
        with incoming('path/to/file.txt') as file:
            # Process the file
            data = file.read()
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

def get_source(line: str) -> str:
    """
    Extracts the source code from a given line of text.
    
    This function searches for a pattern in the input string that matches 
    'Source code: <code>' and extracts the Source code.
    
    Args:
        line (str): A line of text potentially containing the Source code.
    
    Returns:
        str: The extracted source code if the pattern is found, otherwise None.
    """

    source_line = re.compile(r'(Source code:)(\s*)(.*)')
    source_search = source_line.search(line)

    if source_search:
        source_code = source_search.group(3)
    
    return source_code

def get_nhgis(line: str) -> str:
    """
    Extracts the NHGIS code from a given line of text.

    This function searches for a pattern in the input line that matches the format
    'NHGIS code: <code>' and extracts the NHGIS code.

    Args:
        line (str): A line of text that potentially contains the NHGIS code.

    Returns:
        str: The extracted NHGIS code if the pattern is found, otherwise None.
    """

    nhgis_line = re.compile(r'(NHGIS code:)(\s*)(.*)')
    nhgis_search = nhgis_line.search(line)

    if nhgis_search:
        nhgis_code = nhgis_search.group(3)

    return nhgis_code

def get_dict(input_path: str) -> Dict:
    """
    Parses an input file to create a dictionary mapping NHGIS codes to Source codes.
    
    Args:
        input_path (str): The path to the input file to be processed.
    
    Returns:
        dict: A dictionary where the keys are NHGIS codes and the values are Source codes.
    
    Summary:
        The function reads the input file line by line, extracting NHGIS codes and Source codes
        using the `get_source` and `get_nhgis` functions, respectively. If both codes are successfully
        extracted from a line, they are added to the dictionary. The process continues until the end
        of the file is reached.
    """

    dict_cd: Dict = {}
    source_code: str = None
    nhgis_code: str = None

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

def rename_df_columns(df: pd.DataFrame, dict_cd: Dict) -> tuple:
    """
    Renames the columns of a DataFrame based on a provided dictionary of codes.

    Args:
        X_df (pd.DataFrame): The DataFrame whose columns are to be renamed.
        dict_cd (dict): A dictionary where keys are substrings to be replaced in the column names,
                        and values are the corresponding replacements.

    Returns:
        tuple: A tuple containing the list of renamed columns and the DataFrame with renamed columns.
    """
    
    column_list: List = list(df.columns)
    rename_list: List = []

    for index, data in enumerate(column_list):
        for key, value in dict_cd.items():
            if key in data:
                column_list[index] = data.replace(key, value)
                rename_list.append(column_list[index])

    return rename_list, df.set_axis(column_list, axis=1).dropna(axis=1)

def state_to_postal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame of state names to their postal abbreviations.

    This function takes a DataFrame containing state names and returns a new DataFrame 
    with two columns: 'state' and 'state_po'. The 'state' column contains the full 
    names of U.S. states, the District of Columbia, and U.S. territories, while the 
    'state_po' column contains their corresponding postal abbreviations.

    Args:
        df (pd.DataFrame): A DataFrame containing state names.

        pd.DataFrame: A DataFrame with columns 'state' and 'state_po' mapping state 
                      names to their postal abbreviations.

    Returns:
        pd.DataFrame: A DataFrame with state abbreviations and state names.
    """

    us_state_to_abbrev: Dict = {"Alabama": "AL",
                                "Alaska": "AK",
                                "Arizona": "AZ",
                                "Arkansas": "AR",
                                "California": "CA",
                                "Colorado": "CO",
                                "Connecticut": "CT",
                                "Delaware": "DE",
                                "Florida": "FL",
                                "Georgia": "GA",
                                "Hawaii": "HI",
                                "Idaho": "ID",
                                "Illinois": "IL",
                                "Indiana": "IN",
                                "Iowa": "IA",
                                "Kansas": "KS",
                                "Kentucky": "KY",
                                "Louisiana": "LA",
                                "Maine": "ME",
                                "Maryland": "MD",
                                "Massachusetts": "MA",
                                "Michigan": "MI",
                                "Minnesota": "MN",
                                "Mississippi": "MS",
                                "Missouri": "MO",
                                "Montana": "MT",
                                "Nebraska": "NE",
                                "Nevada": "NV",
                                "New Hampshire": "NH",
                                "New Jersey": "NJ",
                                "New Mexico": "NM",
                                "New York": "NY",
                                "North Carolina": "NC",
                                "North Dakota": "ND",
                                "Ohio": "OH",
                                "Oklahoma": "OK",
                                "Oregon": "OR",
                                "Pennsylvania": "PA",
                                "Rhode Island": "RI",
                                "South Carolina": "SC",
                                "South Dakota": "SD",
                                "Tennessee": "TN",
                                "Texas": "TX",
                                "Utah": "UT",
                                "Vermont": "VT",
                                "Virginia": "VA",
                                "Washington": "WA",
                                "West Virginia": "WV",
                                "Wisconsin": "WI",
                                "Wyoming": "WY",
                                "District of Columbia": "DC",
                                "American Samoa": "AS",
                                "Guam": "GU",
                                "Northern Mariana Islands": "MP",
                                "Puerto Rico": "PR",
                                "United States Minor Outlying Islands": "UM",
                                "Virgin Islands, U.S.": "VI"}

    return df['state'].map(us_state_to_abbrev)

def preprocess_poll_data(poll: pd.DataFrame, cycle_year: int) -> pd.DataFrame:
    """
    Preprocesses poll data for a given election cycle year.

    This function filters the poll data to include only the specified election cycle year
    and the two major parties (Republican and Democrat). It then selects the most recent
    poll for each state, district, and party combination. Finally, it calculates the average
    polling percentage for each combination.

    Args:
        poll (pd.DataFrame): The DataFrame containing poll data. It must include the columns
                             'cycle', 'party', 'end_date', 'state_po', 'district', and 'pct'.
        cycle_year (int): The election cycle year to filter the poll data.

    Returns:
        pd.DataFrame: A DataFrame with the average polling percentages for each state, district,
                      and party combination. The DataFrame includes the columns 'state_po',
                      'district', 'party', and 'pct'.
    """

    poll_cycle = poll[(poll['cycle'] == cycle_year) & (poll['party'].isin(['REP', 'DEM']))]
    poll_cycle = poll_cycle[poll_cycle['end_date'] == poll_cycle.groupby(['state_po', 'district', 'party'])['end_date'].transform('max')]
    avg_poll = poll_cycle.groupby(['state_po', 'district', 'party'])['pct'].mean().reset_index()

    return avg_poll

def create_indicator_columns(df: pd.DataFrame, cols: List) -> pd.DataFrame:
    """
    Adds indicator columns to the DataFrame for specified columns.

    This function takes a DataFrame and a list of column names, and creates new columns
    indicating whether the values in the specified columns are null (NaN). The new columns
    will have the same names as the original columns with the suffix '_indicator'. The 
    function then concatenates these indicator columns to the original DataFrame and fills 
    any remaining NaN values with 0.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    cols (List): A list of column names for which to create indicator columns.

    Returns:
    pd.DataFrame: The DataFrame with the added indicator columns.
    """

    return pd.concat([df, pd.DataFrame(df[cols].isnull().astype(int).add_suffix('_indicator'))], axis=1).fillna(0)


#%%
# Load the data
X_train = pd.read_csv(os.path.relpath(f'nhgis0002_ds249_20205_cd116th.csv', start=start), low_memory=False)
X_test = pd.read_csv(os.path.relpath(f'nhgis0001_ds262_20225_cd118th.csv', start=start), low_memory=False)
y = pd.read_csv(os.path.relpath(f'house_outcomes_historical.csv', start=start), low_memory=False)
y['party'] = y['party'].str[:3]
poll = pd.read_csv(os.path.relpath(f'538_house_polls_historical.csv', start=start), low_memory=False).rename(columns={'seat_number':'district'})
poll['state_po'] = state_to_postal(poll)

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
scree = range(1, X_train.shape[1]//base)

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

avg_train = preprocess_poll_data(poll, 2020).merge(y_train[['state_po', 'district', 'GISJOIN']], 
                                                   on=['state_po', 'district']).pivot(index='GISJOIN', columns='party', values='pct').reset_index()
avg_test = preprocess_poll_data(poll, 2022).merge(y_test[['state_po', 'district', 'GISJOIN']], 
                                                  on=['state_po', 'district']).pivot(index='GISJOIN', columns='party', values='pct').reset_index()

y_train = pd.get_dummies(y_train, prefix=['party'], columns=['party']).query('totalvotes > 0')
y_test = pd.get_dummies(y_test, prefix=['party'], columns=['party']).query('totalvotes > 0')

X_train = X_train.merge(avg_train, on='GISJOIN', how='left')
X_test = X_test.merge(avg_test, on='GISJOIN', how='left')

train_set = create_indicator_columns(y_train.merge(X_train, on='GISJOIN'), party_cols)
test_set = create_indicator_columns(y_test.merge(X_test, on='GISJOIN'), party_cols)

#%%
# Export data
train_set.to_csv(os.path.relpath(f'../output/train_set.csv', start=start), index=False)
test_set.to_csv(os.path.relpath(f'../output/test_set.csv', start=start), index=False)

#%%