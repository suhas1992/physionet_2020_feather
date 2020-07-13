#!/usr/bin/env python
# coding: utf-8


import os
import json
import pickle
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from get_12ECG_features import get_12ECG_features
from driver import load_challenge_data

import sklearn
from sklearn.metrics import mutual_info_score

# Define directories here
data_dir = "/share/workhorse3/vsanil/physionet/Training_WFDB"
data_json_path = "/share/workhorse3/vsanil/physionet/"


# Define program variables here
data_dict = {}
dataset_dict = {'S':{'size':549, 'name':'PTB'},
                'HR':{'size':21837, 'name':'PTB-XL'},
                'Q':{'size':3581, 'name':'CPSC_2'},
                'I':{'size':75, 'name':'StPetersburg'},
                'E':{'size':10344, 'name':'Georgia'},
                'A':{'size':6877, 'name':'CPSC_1'}}

# Extracts features and stores as a dictionary
def extract_all_data(path_name, data_dict, filetype):
    # Walk through directory
    count = 0
    for root, dirs, files in os.walk(path_name):
        for idx, filename in enumerate(files):
            if filename.startswith(filetype) and filename.endswith('.mat'):
                count += 1
                print(count, filename)
                data, header = load_challenge_data(os.path.join(root, filename))
                data_dict[filename.replace('.mat','')] = [data, header]
                
    return data_dict

# Save data in pickle format
def save_pickle(path, data_dict, filename):
    with open(os.path.join(path, '{}.p'.format(filename)), 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
# Get label dict
def get_labels(data_dict):
    label_dict = {}
    for filename, feature_list in data_dict.items():
        label = feature_list[1][-4].replace("#Dx: ","").replace("\n","").split(',')
        for l in label:
            if l not in label_dict:
                label_dict[l] = 0
        
    return label_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--filetype", required=True, 
                        help="S:PTB, HR:PTB-XL, Q:CPSC_2, I:StPetersburg, E:Georgia, A:CPSC_1")

    args = parser.parse_args()
    # Check if model dictionary is already present
    
    if os.path.exists(os.path.join(data_json_path, 'data_{}.p'.format(dataset_dict[args.filetype]['name']))):
        with open(os.path.join(data_json_path, 'data_{}.p'.format(dataset_dict[args.filetype]['name'])), 'rb') as fp:
            data_dict = pickle.load(fp)
    else:
        data_dict = extract_all_data(data_dir, data_dict, args.filetype)
        print("Data extracted")
        save_pickle(data_json_path, data_dict, 'data_{}'.format(dataset_dict[args.filetype]['name']))
    
# Generate label dictionary
#label_dict = get_labels(data_dict)
#sex_dict = {"Male":0, "Female":1}


"""
data_dict



obs_dict = {'Age':[], 'Sex':[], 'Hashed Label':[], 'Label':[], 'Feature':[], 'Filename':[]}
feature_hash = {}
count = 0

for filename, feature_list in data_dict.items():
    feat_lead_12 = feature_list[0]
    header_data = feature_list[1]
    ind_obs_l_dict = {}
    
    # Get metadata 
    label = header_data[-4].replace("#Dx: ","").replace("\n","").split(',')
    age = header_data[-6].replace("#Age: ","").replace("\n","").split(',')[0]
    sex = sex_dict[header_data[-5].replace("#Sex: ","").replace("\n","").split(',')[0]]
    
    if age=="NaN":
        continue
    
    # Generate labels
    for k,v in label_dict.items():
        if k in label:
            ind_obs_l_dict[k] = 1
        else:
            ind_obs_l_dict[k] = 0
    feat_label = list(ind_obs_l_dict.values())
    if tuple(feat_label) not in feature_hash:
        feature_hash[tuple(feat_label)] = count
        count += 1
    
    # Construct a observation dictionary
    obs_dict['Age'].append(int(age))
    obs_dict['Sex'].append(sex)
    obs_dict['Hashed Label'].append(feature_hash[tuple(feat_label)])
    obs_dict['Label'].append(feat_label)
    obs_dict['Feature'].append(feat_lead_12)
    obs_dict['Filename'].append(filename)
    
    # Display header
    #for h in header_data:
    #    print(h[:-1])
    
df = pd.DataFrame(obs_dict, columns = list(obs_dict.keys()))


save_pickle(data_json_path, obs_dict,"observations")
"""