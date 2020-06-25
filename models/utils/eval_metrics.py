import config as cfg 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, multilabel_confusion_matrix

def print_conf_mat(true_labels, preds):
    cm = confusion_matrix(true_labels, preds)
    I = pd.Index(['True Negative', 'True Positive'], name="rows")
    C = pd.Index(['Predicted Negative', 'Predicted Positive'], name="columns")
    cm_df = pd.DataFrame(data=cm, index=I, columns=C)

    print("Confusion Matrix \n", cm_df)

def print_multilabel_report(true_labels, preds):
    I = pd.Index(['True Negative', 'True Positive'], name="rows")
    C = pd.Index(['Predicted Negative', 'Predicted Positive'], name="columns")
    ml_cm = multilabel_confusion_matrix(true_labels, preds)
    tot_acc = 0.0 
    tot_prec = 0.0 
    tot_rec = 0.0
    tot_misclass = 0.0

    for i, cm in enumerate(ml_cm):
        print("Confusion Matrix: {}".format(cfg.TARGET_NAMES[i]))
        cm_df = pd.DataFrame(data=cm, index=I, columns=C)

        print(cm_df)

        # Produce the mtrics from confusion matrix
        tp = cm[1,1]
        tn = cm[0,0]
        fn = cm[1,0]
        fp = cm[0,1]

        accuracy = (tp+tn)/np.sum(cm)
        misclass_rate = 1 - accuracy
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        print("Accuracy: ", accuracy, 
              "Misclassification Rate: ", misclass_rate,
              "Recall: ", recall, 
              "Precision: ", precision)

        tot_acc += accuracy
        tot_prec += precision
        tot_rec += recall 
        tot_misclass += misclass_rate

    return tot_acc/i, tot_prec/i, tot_rec/i, tot_misclass/i