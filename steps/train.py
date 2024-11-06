import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import xgboost

def train(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> xgboost.XGBClassifier:
    """
    Trains an XGBoost classifier on the provided training data and evaluates it on the validation data.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
    
    Returns:
        xgboost.XGBClassifier: The trained XGBoost classifier model.
    
    Summary:
        1. Prints the shapes of the training and validation datasets.
        2. Initializes an XGBoost classifier with specified hyperparameters.
        3. Trains the model on the training data and evaluates it on the validation data with early stopping.
        4. Prints evaluation metrics (accuracy, precision, recall, AUC) for both the validation and training datasets.
        5. Saves the trained model to a specified file path.
    """

    min_child_weight: int = 8
    max_bin: int = 48
    num_parallel_tree: int = 96
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bynode: float = 0.8
    verbose: bool = False

    print('Train features shape: {}'.format(X_train.shape))
    print('Train labels shape: {}'.format(y_train.shape))
    print('Validation features shape: {}'.format(X_val.shape))
    print('Validation labels shape: {}'.format(y_val.shape))

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
                                                            use_label_encoder=False, 
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

    model = xgboost.XGBClassifier(tree_method='hist', 
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
                                  use_label_encoder=False, 
                                  n_jobs=-1).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_val = model.predict(X_val)
    
    print ("Metrics for validation set")
    print('')
    print (pd.crosstab(index=y_val, columns=np.round(pred_val),
                       rownames=['Actuals'], colnames=['Predictions'], margins=True))
    print('')

    pred_val = np.round(pred_val)

    val_accuracy = accuracy_score(y_val, pred_val)
    val_precision = precision_score(y_val, pred_val)
    val_recall = recall_score(y_val, pred_val)

    print("Accuracy Model (Validation): %.2f%%" % (val_accuracy * 100.0))
    print("Precision Model (Validation): %.2f" % (val_precision))
    print("Recall Model (Validation): %.2f" % (val_recall))

    val_auc = roc_auc_score(y_val, pred_val)
    print("Model AUC (Validation): %.2f" % (val_auc))

    pred_train = model.predict(X_train)

    print ("Metrics for training set")
    print('')
    print (pd.crosstab(index=y_train, columns=np.round(pred_train),
                       rownames=['Actuals'], colnames=['Predictions'], margins=True))
    print('')

    pred_train = np.round(pred_train)

    train_accuracy = accuracy_score(y_train, pred_train)
    train_precision = precision_score(y_train, pred_train)
    train_recall = recall_score(y_train, pred_train)

    print("Accuracy Model (Training): %.2f%%" % (train_accuracy * 100.0))
    print("Precision Model (Training): %.2f" % (train_precision))
    print("Recall Model (Training): %.2f" % (train_recall))

    train_auc = roc_auc_score(y_train, pred_train)
    print("Model AUC (Training): %.2f" % (train_auc))

    model_file_path="/opt/ml/model/xgboost_model.bin"
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    model.save_model(model_file_path)

    return model
