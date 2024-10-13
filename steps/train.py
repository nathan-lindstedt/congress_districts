import os
import shutil

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import xgboost

def train(X_train, y_train, X_val, y_val):

    print('Train features shape: {}'.format(X_train.shape))
    print('Train labels shape: {}'.format(y_train.shape))
    print('Validation features shape: {}'.format(X_val.shape))
    print('Validation labels shape: {}'.format(y_val.shape))

    xgb = xgboost.XGBClassifier(n_estimators=500,
                                gamma=0, 
                                learning_rate=0.05, 
                                max_depth=3, 
                                eval_metric='auc', 
                                colsample_bytree=0.8, 
                                colsample_bylevel=0.8, 
                                colsample_bynode=0.8, 
                                num_parallel_tree=48)

    model = xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)

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
