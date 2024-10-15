import os
import joblib

import pandas as pd

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess(input_data_s3_uri: str) -> Tuple:
    """
    Preprocesses the input data for training, validation, and testing.

    This function reads training and testing datasets from the specified S3 URI, 
    splits the testing data into validation and testing sets, applies transformations 
    to the features, and saves the transformation model. It returns the processed 
    features and labels for training, validation, and testing, along with the 
    transformation model.

    Args:
        input_data_s3_uri (str): The S3 URI where the input data is stored.

    Returns:
        tuple: A tuple containing the following elements:
            - X_train (numpy.ndarray): The preprocessed training features.
            - y_train (numpy.ndarray): The training labels.
            - X_val (numpy.ndarray): The preprocessed validation features.
            - y_val (numpy.ndarray): The validation labels.
            - X_test (numpy.ndarray): The preprocessed testing features.
            - y_test (numpy.ndarray): The testing labels.
            - featurizer_model (ColumnTransformer): The fitted transformation model.
    """

    data_columns: List = ['0', '1', '2', '3', '4', '5']
    target_column: str = 'party_REP'

    train_set = pd.read_csv(f"{input_data_s3_uri}/train_set.csv")
    test_set = pd.read_csv(f"{input_data_s3_uri}/test_set.csv")

    X_train = train_set.drop(target_column, axis=1)
    X_train = X_train[data_columns]
    y_train = train_set[target_column]

    X_test = test_set.drop(target_column, axis=1)
    X_test = X_test[data_columns]
    y_test = test_set[target_column]   

    validation_ratio = 0.1

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=validation_ratio, random_state=2)

    # Apply transformations
    transformer = ColumnTransformer(transformers=[
                                                # ('numeric', StandardScaler(), data_columns),
                                                # ('categorical', OneHotEncoder(), data_columns)
                                                 ],
                                    remainder='passthrough')
    featurizer_model = transformer.fit(X_train)
    X_train = featurizer_model.transform(X_train)
    X_val = featurizer_model.transform(X_val)

    print(f'Shape of train features after preprocessing: {X_train.shape}')
    print(f'Shape of validation features after preprocessing: {X_val.shape}')
    print(f'Shape of test features after preprocessing: {X_test.shape}\n')

    y_train = y_train.values.reshape(-1)
    y_val = y_val.values.reshape(-1)

    print(f'Shape of train labels after preprocessing: {y_train.shape}')
    print(f'Shape of validation labels after preprocessing: {y_val.shape}')
    print(f'Shape of test labels after preprocessing: {y_test.shape}')

    model_file_path="/opt/ml/model/sklearn_model.joblib"
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    joblib.dump(featurizer_model, model_file_path)

    return X_train, y_train, X_val, y_val, X_test, y_test, featurizer_model
