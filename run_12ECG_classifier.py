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
import ast

# Language defined keyword
keyword = 'pytorch'
drive = False
with open("checks.txt", 'r') as f:
    for lines in f:
        if "False" in lines:
            drive = True
            break
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_pytorch(input_directory):
    tempdir = ""
    if drive:
        file_id = "1ONP9cPL93MGqmtMbPDwZ_SBn7Z69iJss"
        tempdir = tempfile.mkdtemp()
        filepath = os.path.join(tempdir,'best_model.pth')
        f = open(filepath, 'wb')
        download_file_from_google_drive(file_id, filepath)
        print("Model receieved!")
    else:
        filepath = input_directory

    # Define the model here
    output_dim = 27
    model = ResNet(BasicBlock, [2,2,2,2], 
                   num_classes=output_dim, 
                   groups=32, 
                   width_per_group=4)
    model.to(device)
    
    # Load the model
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if drive:
        shutil.rmtree(tempdir)
        
    return model

def classify_pytorch(data, model):
    model.eval()
    data = torch.from_numpy(data).float().unsqueeze(0).to(device)
    preds = model(data)

    return preds[0].cpu().detach().numpy()

# Dictionary maintaing relevant framework related funcs
lang_dict = {'pytorch':{'model':load_pytorch, 'classify':classify_pytorch}}

def run_12ECG_classifier(data,header_data,model,num_classes = 27):

    label = np.zeros(num_classes, dtype=int)
    score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class.
    preds = lang_dict[keyword]['classify'](data, model)

    label = np.where(preds > 0.5, 1, 0)
    score = preds
    
    return label, score

def load_12ECG_model(input_directory):
    # load the model from disk 
    loaded_model = lang_dict[keyword]['model'](input_directory)

    # Change file config to false
    vals = []
    with open("checks.txt", "r") as fr:
        for lines in fr:
            vals.append(fr)

    vals[0] = "False"
    with open("checks.txt", "w") as f:
        for v in vals:
            f.write("{}\n".format(v))

    return loaded_model
