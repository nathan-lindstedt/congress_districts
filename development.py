#%%
# Import libraries
import joblib
import os
import time
import urllib

import numpy as np
import pandas as pd
import sklearn

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import xgboost

#%%
data_columns: List = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
target_column: str = 'party_REP'

train_set = pd.read_csv(f"/Users/nathan.lindstedt/Documents/GitHub Projects/congress_districts/output/train_set.csv")
test_set = pd.read_csv(f"/Users/nathan.lindstedt/Documents/GitHub Projects/congress_districts/output/test_set.csv")

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
                                            #   ('numeric', StandardScaler(), data_columns),
                                            #   ('categorical', OneHotEncoder(), data_columns)
                                            ],
                                remainder='passthrough')
featurizer_model = transformer.fit(X_train)
X_train = featurizer_model.transform(X_train)
X_val = featurizer_model.transform(X_val)
X_test = featurizer_model.transform(X_test)

print(f'Shape of train features after preprocessing: {X_train.shape}')
print(f'Shape of validation features after preprocessing: {X_val.shape}')
print(f'Shape of test features after preprocessing: {X_test.shape}\n')

y_train = y_train.values.reshape(-1)
y_val = y_val.values.reshape(-1)

print(f'Shape of train labels after preprocessing: {y_train.shape}')
print(f'Shape of validation labels after preprocessing: {y_val.shape}')
print(f'Shape of test labels after preprocessing: {y_test.shape}')

#%%
min_child_weight: int = 4
max_bin: int = 48
num_parallel_tree: int = 96
subsample: float = 0.8
colsample_bytree: float = 0.8
colsample_bynode: float = 0.8
verbose: bool = True

xgb_start = time.perf_counter()

xgb_hyperparameters = [{'max_depth': np.linspace(1, 25, 25, dtype=int, endpoint=True),
                        'gamma': np.linspace(0, 1, 10, dtype=float, endpoint=True),
                        'learning_rate': [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]}]

xgb_gridsearch = HalvingGridSearchCV(xgboost.XGBClassifier(tree_method='hist', 
                                                           grow_policy='depthwise', 
                                                           min_child_weight=min_child_weight, 
                                                           max_bin=max_bin, 
                                                           num_parallel_tree=num_parallel_tree, 
                                                           subsample=subsample, 
                                                           colsample_bytree=colsample_bytree, 
                                                           colsample_bynode=colsample_bynode, 
                                                           eval_metric='logloss',
                                                           early_stopping_rounds=20, 
                                                           n_jobs=-1), 
                                                           xgb_hyperparameters,
                                                           scoring='roc_auc',
                                                           resource='n_estimators', 
                                                           factor=3, 
                                                           min_resources=5, 
                                                           max_resources=1000, 
                                                           aggressive_elimination=True,  
                                                           cv=3, 
                                                           verbose=verbose, 
                                                           n_jobs=-1)

xgb_best_model = xgb_gridsearch.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_stop = time.perf_counter()

print(f'XGB Random Forest model trained in {(xgb_stop - xgb_start)/60:.1f} minutes')
print(f'Best XGB Random Forest parameters: {xgb_gridsearch.best_params_}')

#%%
xgb_xgbrf = xgboost.XGBClassifier(tree_method='hist', 
                          grow_policy='depthwise', 
                          min_child_weight=min_child_weight, 
                          max_bin=max_bin, 
                          num_parallel_tree=num_parallel_tree, 
                          subsample=subsample, 
                          colsample_bytree=colsample_bytree, 
                          colsample_bynode=colsample_bynode, 
                          eval_metric='logloss',
                          early_stopping_rounds=20,  
                          **xgb_gridsearch.best_params_, 
                          n_jobs=-1).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

#%%
pred_train = xgb_xgbrf.predict(X_train)
pred_val = xgb_xgbrf.predict(X_val)
pred_test = xgb_xgbrf.predict(X_test)

#%%
print ("Metrics for training set")
print('')
print (pd.crosstab(index=y_train, columns=np.round(pred_train),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

train_accuracy = accuracy_score(y_train, pred_train)
train_precision = precision_score(y_train, pred_train)
train_recall = recall_score(y_train, pred_train)

print("Accuracy Model (Training): %.1f%%" % (train_accuracy * 100.0))
print("Precision Model (Training): %.2f" % (train_precision))
print("Recall Model (Training): %.2f" % (train_recall))

train_auc = roc_auc_score(y_train, pred_train)
print("Model AUC (Training): %.2f" % (train_auc))

print('')

print ("Metrics for validation set")
print('')
print (pd.crosstab(index=y_val, columns=np.round(pred_val),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

val_accuracy = accuracy_score(y_val, pred_val)
val_precision = precision_score(y_val, pred_val)
val_recall = recall_score(y_val, pred_val)

print("Accuracy Model (Validation): %.1f%%" % (val_accuracy * 100.0))
print("Precision Model (Validation): %.2f" % (val_precision))
print("Recall Model (Validation): %.2f" % (val_recall))

val_auc = roc_auc_score(y_val, pred_val)
print("Model AUC (Validation): %.2f" % (val_auc))

print('')
print ("Metrics for testing set")
print('')
print (pd.crosstab(index=y_test, columns=np.round(pred_test),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

test_accuracy = accuracy_score(y_test, pred_test)
test_precision = precision_score(y_test, pred_test)
test_recall = recall_score(y_test, pred_test)

print("Accuracy Model (Testing): %.1f%%" % (test_accuracy * 100.0))
print("Precision Model (Testing): %.2f" % (test_precision))
print("Recall Model (Testing): %.2f" % (test_recall))

test_auc = roc_auc_score(y_test, pred_test)
print("Model AUC (Testing): %.2f" % (test_auc))

#%%