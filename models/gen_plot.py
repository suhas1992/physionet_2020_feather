import os
import sys
sys.path.append("models")

import csv
import math 
import pickle 
import argparse 
import main as mn 
import data as dt
import config as cfg 
import numpy as np 
from scipy.io import loadmat
import utils.get_class_dict as cdict

import torch
import torch.nn as nn
from torchsummary import summary 
from networks.mobile import mobileNet
from networks.resnext import ResNet, Bottleneck, BasicBlock

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data

# Extract challenge data for a particular group
def extract_challenge_data(count, group_name):
    invalid_count = 0
    count = 0

    feature_dict = {'features':[], 'labels':[]}
    labels = [0 for _ in CLASS_DICT[group]]
    keys = {k:idx for idx, k in enumerate(CLASS_DICT[group])}

    for idx, f in enumerate(files):
        if f.endswith('.mat'):
            #print(count, f, idx)
            if count % 1000 == 0:
                print("RAM used: ", psutil.virtual_memory().percent, "Files done: ", count)
            data, header = load_challenge_data(f)
            label = header[-4].replace("#Dx: ","").replace("\n","").split(',')
            l = labels.copy()
            add = False 
            for lbl in label:
                try:
                    l[keys[lbl]] = 1
                    add = True
                except KeyError:
                    continue
            if add:
                feature_dict['features'].append(data)
                feature_dict['labels'].append(l) 
            count += 1

    print("Features for group {} extracted".format(group))
    return feature_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument("-i", "--datadir", required=True,
                        help="Input directory")
    parser.add_argument("-e", "--examples", required=False, default="10",
                        help="Number of examples")
    parser.add_argument("-g","--group", required=False, default="0",
                        help="Group number(1-7)")
    parser.add_argument("d", "--diagnosis", required=True,
                        help="Refer to Dx mapping csv to get a list of diagnosis")
    args = parser.parse_args() 

    if not os.path.exists(args.datadir):
        print("Path does not exist")
        exit()

    snomed, group_ls = cfg.get_class_group(args.diagnosis)


