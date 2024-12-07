#%%
# Import libraries
import os
import time

import numpy as np
import pandas as pd

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
# System path variables
start: str = os.path.dirname(__file__)

#%%
# Load the data, split into training and testing sets, and preprocess
model_one_columns: List = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
model_two_columns: List = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'REP', 'REP_indicator']

target_column: str = 'party_REP'
drop_column: str = 'REP_indicator'

train_set = pd.read_csv(os.path.relpath(f"../congress_districts/output/train_set.csv", start=start), low_memory=False)
test_set = pd.read_csv(os.path.relpath(f"../congress_districts/output/test_set.csv", start=start), low_memory=False)

model_one_X_train = train_set.drop(target_column, axis=1)
model_one_X_train = model_one_X_train[model_one_columns]
model_one_y_train = train_set[target_column]

model_two_X_train = train_set.drop(target_column, axis=1)
model_two_X_train = model_two_X_train[model_two_columns]
model_two_y_train = train_set[target_column]

model_one_X_test = test_set.drop(target_column, axis=1)
model_one_X_test = model_one_X_test[model_one_columns]
model_one_y_test = test_set[target_column]

model_two_X_test = test_set.drop(target_column, axis=1)
model_two_X_test = model_two_X_test[model_two_columns]
model_two_y_test = test_set[target_column]

validation_ratio = 0.1

model_one_X_train, model_one_X_val, model_one_y_train, model_one_y_val = train_test_split(model_one_X_train, model_one_y_train, test_size=validation_ratio, random_state=2)
model_two_X_train, model_two_X_val, model_two_y_train, model_two_y_val = train_test_split(model_two_X_train, model_two_y_train, test_size=validation_ratio, random_state=2)

model_one_transformer = ColumnTransformer(transformers=[
                                            #   ('numeric', StandardScaler(), model_one_columns),
                                            #   ('categorical', OneHotEncoder(), model_one_columns)
                                            ],
                                remainder='passthrough')
model_one_featurizer = model_one_transformer.fit(model_one_X_train)
model_one_X_train = model_one_featurizer.transform(model_one_X_train)
model_one_X_val = model_one_featurizer.transform(model_one_X_val)
model_one_X_test = model_one_featurizer.transform(model_one_X_test)

print(f'Shape of model one train features after preprocessing: {model_one_X_train.shape}')
print(f'Shape of model one validation features after preprocessing: {model_one_X_val.shape}')
print(f'Shape of model one test features after preprocessing: {model_one_X_test.shape}\n')

model_one_y_train = model_one_y_train.values.reshape(-1)
model_one_y_val = model_one_y_val.values.reshape(-1)

print(f'Shape of model one train labels after preprocessing: {model_one_y_train.shape}')
print(f'Shape of model one validation labels after preprocessing: {model_one_y_val.shape}')
print(f'Shape of model one test labels after preprocessing: {model_one_y_test.shape}\n')

model_two_transformer = ColumnTransformer(transformers=[
                                            #   ('numeric', StandardScaler(), model_two_columns),
                                            #   ('categorical', OneHotEncoder(), model_two_columns)
                                            ],
                                remainder='passthrough')
model_two_featurizer = model_two_transformer.fit(model_two_X_train)
model_two_X_train = model_two_featurizer.transform(model_two_X_train)
model_two_X_val = model_two_featurizer.transform(model_two_X_val)
model_two_X_test = model_two_featurizer.transform(model_two_X_test)

print(f'Shape of model two train features after preprocessing: {model_two_X_train.shape}')
print(f'Shape of model two validation features after preprocessing: {model_two_X_val.shape}')
print(f'Shape of model two test features after preprocessing: {model_two_X_test.shape}\n')

model_two_y_train = model_two_y_train.values.reshape(-1)
model_two_y_val = model_two_y_val.values.reshape(-1)

print(f'Shape of model two train labels after preprocessing: {model_two_y_train.shape}')
print(f'Shape of model two validation labels after preprocessing: {model_two_y_val.shape}')
print(f'Shape of model two test labels after preprocessing: {model_two_y_test.shape}')

#%%
# Train the models with hyperparameter tuning
min_child_weight: int = 5
max_bin: int = 48
num_parallel_tree: int = 96
subsample: float = 0.8
colsample_bytree: float = 0.8
colsample_bynode: float = 0.8
verbose: bool = True

xgb_hyperparameters = [{'max_depth': np.linspace(1, 25, 25, dtype=int, endpoint=True),
                        'gamma': np.linspace(0, 1, 10, dtype=float, endpoint=True),
                        'learning_rate': [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]}]

model_one_xgb_start = time.perf_counter()

model_one_xgb_gridsearch = HalvingGridSearchCV(xgboost.XGBClassifier
                                              (tree_method='hist', 
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

model_one_xgb_best_model = model_one_xgb_gridsearch.fit(model_one_X_train, model_one_y_train, eval_set=[(model_one_X_val, model_one_y_val)], verbose=False)

model_one_xgb_stop = time.perf_counter()

print(f'\nXGB Random Forest model one trained in {(model_one_xgb_stop - model_one_xgb_start)/60:.1f} minutes')
print(f'Best XGB Random Forest model one parameters: {model_one_xgb_gridsearch.best_params_}\n')

model_two_xgb_start = time.perf_counter()

model_two_xgb_gridsearch = HalvingGridSearchCV(xgboost.XGBClassifier
                                              (tree_method='hist', 
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

model_two_xgb_best_model = model_two_xgb_gridsearch.fit(model_two_X_train, model_two_y_train, eval_set=[(model_two_X_val, model_two_y_val)], verbose=False)

model_two_xgb_stop = time.perf_counter()

print(f'\nXGB Random Forest model two trained in {(model_two_xgb_stop - model_two_xgb_start)/60:.1f} minutes')
print(f'Best XGB Random Forest model two parameters: {model_two_xgb_gridsearch.best_params_}\n')

#%%
# Train the models with the best hyperparameters and make predictions
model_one_xgb_xgbrf = xgboost.XGBClassifier(tree_method='hist', 
                                            grow_policy='depthwise', 
                                            min_child_weight=min_child_weight, 
                                            max_bin=max_bin, 
                                            num_parallel_tree=num_parallel_tree, 
                                            subsample=subsample, 
                                            colsample_bytree=colsample_bytree, 
                                            colsample_bynode=colsample_bynode, 
                                            eval_metric='logloss',
                                            early_stopping_rounds=20,  
                                            **model_one_xgb_gridsearch.best_params_, 
                                            n_jobs=-1).fit(model_one_X_train, model_one_y_train, eval_set=[(model_one_X_val, model_one_y_val)], verbose=False)

model_two_xgb_xgbrf = xgboost.XGBClassifier(tree_method='hist', 
                                            grow_policy='depthwise', 
                                            min_child_weight=min_child_weight, 
                                            max_bin=max_bin, 
                                            num_parallel_tree=num_parallel_tree, 
                                            subsample=subsample, 
                                            colsample_bytree=colsample_bytree, 
                                            colsample_bynode=colsample_bynode, 
                                            eval_metric='logloss',
                                            early_stopping_rounds=20,  
                                            **model_two_xgb_gridsearch.best_params_, 
                                            n_jobs=-1).fit(model_two_X_train, model_two_y_train, eval_set=[(model_two_X_val, model_two_y_val)], verbose=False)

model_one_pred_train = model_one_xgb_xgbrf.predict(model_one_X_train)
model_one_pred_val = model_one_xgb_xgbrf.predict(model_one_X_val)
model_one_pred_test = model_one_xgb_xgbrf.predict(model_one_X_test)

model_two_pred_train = model_two_xgb_xgbrf.predict(model_two_X_train)
model_two_pred_val = model_two_xgb_xgbrf.predict(model_two_X_val)
model_two_pred_test = model_two_xgb_xgbrf.predict(model_two_X_test)

#%%
# Print the results for model one w/o polling data
print ("Metrics for training set")
print('')
print (pd.crosstab(index=model_one_y_train, columns=np.round(model_one_pred_train),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

model_one_train_accuracy = accuracy_score(model_one_y_train, model_one_pred_train)
model_one_train_precision = precision_score(model_one_y_train, model_one_pred_train)
model_one_train_recall = recall_score(model_one_y_train, model_one_pred_train)

print("Accuracy Model (Training): %.1f%%" % (model_one_train_accuracy * 100.0))
print("Precision Model (Training): %.2f" % (model_one_train_precision))
print("Recall Model (Training): %.2f" % (model_one_train_recall))

model_one_train_auc = roc_auc_score(model_one_y_train, model_one_pred_train)
print("Model AUC (Training): %.2f" % (model_one_train_auc))

print('')

print ("Metrics for validation set")
print('')
print (pd.crosstab(index=model_one_y_val, columns=np.round(model_one_pred_val),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

model_one_val_accuracy = accuracy_score(model_one_y_val, model_one_pred_val)
model_one_val_precision = precision_score(model_one_y_val, model_one_pred_val)
model_one_val_recall = recall_score(model_one_y_val, model_one_pred_val)

print("Accuracy Model (Validation): %.1f%%" % (model_one_val_accuracy * 100.0))
print("Precision Model (Validation): %.2f" % (model_one_val_precision))
print("Recall Model (Validation): %.2f" % (model_one_val_recall))

model_one_val_auc = roc_auc_score(model_one_y_val, model_one_pred_val)
print("Model AUC (Validation): %.2f" % (model_one_val_auc))

print('')
print ("Metrics for testing set")
print('')
print (pd.crosstab(index=model_one_y_test, columns=np.round(model_one_pred_test),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

model_one_test_accuracy = accuracy_score(model_one_y_test, model_one_pred_test)
model_one_test_precision = precision_score(model_one_y_test, model_one_pred_test)
model_one_test_recall = recall_score(model_one_y_test, model_one_pred_test)

print("Accuracy Model (Testing): %.1f%%" % (model_one_test_accuracy * 100.0))
print("Precision Model (Testing): %.2f" % (model_one_test_precision))
print("Recall Model (Testing): %.2f" % (model_one_test_recall))

model_one_test_auc = roc_auc_score(model_one_y_test, model_one_pred_test)
print("Model AUC (Testing): %.2f" % (model_one_test_auc))

#%%
# Print the results for model two w/ polling data
print ("Metrics for training set")
print('')
print (pd.crosstab(index=model_two_y_train, columns=np.round(model_two_pred_train),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

model_two_train_accuracy = accuracy_score(model_two_y_train, model_two_pred_train)
model_two_train_precision = precision_score(model_two_y_train, model_two_pred_train)
model_two_train_recall = recall_score(model_two_y_train, model_two_pred_train)

print("Accuracy Model (Training): %.1f%%" % (model_two_train_accuracy * 100.0))
print("Precision Model (Training): %.2f" % (model_two_train_precision))
print("Recall Model (Training): %.2f" % (model_two_train_recall))

model_two_train_auc = roc_auc_score(model_two_y_train, model_two_pred_train)
print("Model AUC (Training): %.2f" % (model_two_train_auc))

print('')

print ("Metrics for validation set")
print('')
print (pd.crosstab(index=model_two_y_val, columns=np.round(model_two_pred_val),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

model_two_val_accuracy = accuracy_score(model_two_y_val, model_two_pred_val)
model_two_val_precision = precision_score(model_two_y_val, model_two_pred_val)
model_two_val_recall = recall_score(model_two_y_val, model_two_pred_val)

print("Accuracy Model (Validation): %.1f%%" % (model_two_val_accuracy * 100.0))
print("Precision Model (Validation): %.2f" % (model_two_val_precision))
print("Recall Model (Validation): %.2f" % (model_two_val_recall))

model_two_val_auc = roc_auc_score(model_two_y_val, model_two_pred_val)
print("Model AUC (Validation): %.2f" % (model_two_val_auc))

print('')
print ("Metrics for testing set")
print('')
print (pd.crosstab(index=model_two_y_test, columns=np.round(model_two_pred_test),
                    rownames=['Actuals'], colnames=['Predictions'], margins=True))
print('')

model_two_test_accuracy = accuracy_score(model_two_y_test, model_two_pred_test)
model_two_test_precision = precision_score(model_two_y_test, model_two_pred_test)
model_two_test_recall = recall_score(model_two_y_test, model_two_pred_test)

print("Accuracy Model (Testing): %.1f%%" % (model_two_test_accuracy * 100.0))
print("Precision Model (Testing): %.2f" % (model_two_test_precision))
print("Recall Model (Testing): %.2f" % (model_two_test_recall))

model_two_test_auc = roc_auc_score(model_two_y_test, model_two_pred_test)
print("Model AUC (Testing): %.2f" % (model_two_test_auc))

#%%