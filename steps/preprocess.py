import os
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import DataConversionWarning

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def preprocess(input_data_s3_uri: str) -> tuple :

    columns_cd116 = ['AMZME002', 'AMZME003', 'AMZME004', 
                        'AMZME005', 'AMP4E003', 'AMUQE002', 'AMUQE003', 'AMUQE004',
                        'AMUQE005', 'AMQLE002', 'AMQLE015', 'AMQRE002', 'AMQRE005',
                        'AMRJE002', 'AMRYE015', 'AMZLE003', 'AMRYE016', 'AMRYE017', 
                        'AMRYE018', 'AMRYE032', 'AMRYE033', 'AMRYE034', 'AMRYE035',
                        'AMZLE009', 'AMZLE012', 'AMSPE002', 'AMSQE002', 'AMSTE002',
                        'AMZNE005', 'AMZNE009', 'AMSOE002']
                        
    columns_cd118 = ['AQPZE002', 'AQPZE003', 'AQPZE004', 
                        'AQPZE005', 'AQNPE003', 'AQS1M002', 'AQS1M003', 'AQS1M004', 
                        'AQS1M005', 'AQN6E002', 'AQN6E015', 'AQOCE002', 'AQOCE005', 
                        'AQO4E002', 'AQPJE015', 'AQPJE016', 'AQPJE017', 'AQPJE018',
                        'AQPJE032', 'AQPJE033', 'AQPJE034', 'AQPJE035', 'AQPXE003', 
                        'AQPXE009', 'AQPXE012', 'AQQNE002', 'AQQOE002', 'AQQRE002', 
                        'AQR0E005', 'AQR0E009', 'AQQME002']

    num_columns_cd116 = ['AMZME002', 'AMZME003', 'AMZME004', 
                            'AMZME005', 'AMP4E003', 'AMUQE002', 'AMUQE003', 'AMUQE004',
                            'AMUQE005', 'AMQLE002', 'AMQLE015', 'AMQRE002', 'AMQRE005',
                            'AMRJE002', 'AMRYE015', 'AMZLE003', 'AMRYE016', 'AMRYE017', 
                            'AMRYE018', 'AMRYE032', 'AMRYE033', 'AMRYE034', 'AMRYE035',
                            'AMZLE009', 'AMZLE012', 'AMSPE002', 'AMSQE002', 'AMSTE002',
                            'AMZNE005', 'AMZNE009', 'AMSOE002']

    num_columns_cd118 = ['AQPZE002', 'AQPZE003', 'AQPZE004', 
                            'AQPZE005', 'AQNPE003', 'AQS1M002', 'AQS1M003', 'AQS1M004', 
                            'AQS1M005', 'AQN6E002', 'AQN6E015', 'AQOCE002', 'AQOCE005', 
                            'AQO4E002', 'AQPJE015', 'AQPJE016', 'AQPJE017', 'AQPJE018',
                            'AQPJE032', 'AQPJE033', 'AQPJE034', 'AQPJE035', 'AQPXE003', 
                            'AQPXE009', 'AQPXE012', 'AQQNE002', 'AQQOE002', 'AQQRE002', 
                            'AQR0E005', 'AQR0E009', 'AQQME002']

    target_column = 'party_REPUBLICAN'

    train_set = pd.read_csv(f"{input_data_s3_uri}/train_set.csv")
    test_set = pd.read_csv(f"{input_data_s3_uri}/test_set.csv")

    X_train = train_set.drop(target_column, axis=1)
    X_train = X_train[columns_cd116]
    y_train = train_set[target_column]

    X_test = test_set.drop(target_column, axis=1)
    X_test = X_test[columns_cd118]
    y_test = test_set[target_column]   

    validation_ratio = 0.1

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_ratio, random_state=2)

    # Apply transformations
    transformer = ColumnTransformer(transformers=[
                                                # ('numeric', StandardScaler(), num_columns_cd116),
                                                # ('categorical', OneHotEncoder(), cat_columns_cd116)
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
