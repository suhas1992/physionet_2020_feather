import config as cfg 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, multilabel_confusion_matrix

def print_conf_mat(true_labels, preds):
    cm = confusion_matrix(true_labels, preds)
    I = pd.Index(['True Positive', 'True Negative'], name="rows")
    C = pd.Index(['Predicted Positive', 'Predicted Negative'], name="columns")
    cm_df = pd.DataFrame(data=cm, index=I, columns=C)

    print("Confusion Matrix \n", cm_df)

def print_multilabel_conf_mat(true_labels, preds):
    I = pd.Index(['True Positive', 'True Negative'], name="rows")
    C = pd.Index(['Predicted Positive', 'Predicted Negative'], name="columns")
    ml_cm = multilabel_confusion_matrix(true_labels, preds)

    for i, cm in enumerate(ml_cm):
        print("Confusion Matrix: {}".format(cfg.TARGET_NAMES[i]))
        cm_df = pd.DataFrame(data=cm, index=I, columns=C)

        print(cm_df)
