#!/usr/bin/env python

import numpy as np
import os
import joblib
import shutil 
import tempfile 
import torch
from download import download_file_from_google_drive
from models.networks.resnext import ResNet, BasicBlock
from get_12ECG_features import get_12ECG_features

# Language defined keyword
keyword = 'pytorch'
drive = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_pytorch():
    if drive:
        file_id = "125FLLN21q4dvFGlIWTwUN83BdeGkE5Un"
        tempdir = tempfile.mkdtemp()
        filepath = os.path.join(tempdir,'best_model.pth')
        f = open(filepath, 'wb')
        download_file_from_google_drive(file_id, filepath)
    else:
        filepath = '/home/vsanil/workhorse3/physionet/best_models/best_model.pth'

    # Define the model here
    output_dim = 9
    model = ResNet(BasicBlock, [2,2,2,2], 
                   num_classes=output_dim, 
                   groups=32, 
                   width_per_group=4)
    model.to(device)
    
    # Load the model
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def classify_pytorch(data, model):
    model.eval()
    data = torch.from_numpy(data).float().unsqueeze(0).to(device)
    preds = model(data)

    return preds[0].cpu().detach().numpy()

# Dictionary maintaing relevant framework related funcs
lang_dict = {'pytorch':{'model':load_pytorch, 'classify':classify_pytorch}}

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    label = np.zeros(num_classes, dtype=int)
    score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)

    preds = lang_dict[keyword]['classify'](data, model)

    label = np.where(preds > 0.5, 1, 0)
    score = preds

    return label, score

def load_12ECG_model():
    # load the model from disk 
    loaded_model = lang_dict[keyword]['model']()

    return loaded_model