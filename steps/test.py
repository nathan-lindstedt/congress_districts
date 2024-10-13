import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import xgboost

def test(featurizer_model, booster, X_test, y_test):

    # X_test = featurizer_model.transform(X_test)
    y_test = y_test.values.reshape(-1)

    test_pred = booster.predict(X_test)
    
    print('')
    print (pd.crosstab(index=y_test, columns=np.round(test_pred), 
                                     rownames=['Actuals'], 
                                     colnames=['Predictions'], 
                                     margins=True))
    print('')

    test_pred = np.round(test_pred)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    print('')

    print("Accuracy Model A: %.2f%%" % (test_accuracy * 100.0))
    print("Precision Model A: %.2f" % (test_precision))
    print("Recall Model A: %.2f" % (test_recall))

    test_auc = roc_auc_score(y_test, test_pred)
    print("AUC A: %.2f" % (test_auc))

    report_dict = {
        "binary_classification_metrics": {
            "recall": {"value": test_recall, "standard_deviation": ""},
            "precision": {"value": test_precision, "standard_deviation": ""},
            "accuracy": {"value": test_accuracy, "standard_deviation": ""},
            "auc": {"value": test_auc, "standard_deviation": ""},
        }
    }
    print(f"evaluation report: {report_dict}")

    return report_dict