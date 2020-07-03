#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features

keyword = 'pytorch'
lang_dict = {'pytorch':{'model':load_pytorch}}

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    label = np.zeros(num_classes, dtype=int)
    score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)

    prob_score = model.predict_proba(feats_reshape)

    max_prob_1 = 0
    max_prob_index = 0
    for i, probs in enumerate(prob_score):
        prob_0 = probs[0][0]
        prob_1 = probs[0][1]
        
        score[i] = prob_1
        if prob_1 > 0.5:
            label[i] = 1
        if prob_1 > max_prob_1:
            max_prob_1 = prob_1
            max_prob_index = i
    label[max_prob_index] = 1
    # label = model.predict(feats_reshape)
    # score = model.predict_proba(feats_reshape)

    # current_label[label] = 1

    # for i in range(num_classes):
    #     current_score[i] = np.array(score[0][i])

    return label, score

def load_12ECG_model():
    # load the model from disk 
    filename='randomforest_model.sav'
    loaded_model = joblib.load(filename)

    return loaded_model

def load_pytorch(path):
    filename="/share/workhorse3/vsanil/"